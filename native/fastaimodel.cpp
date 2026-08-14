#include <jni.h>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "llama.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct FastAIModelHandle {
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;
};

static bool g_verbose = false;

static void quiet_log_callback(ggml_log_level level, const char* text, void* user_data) {
    if (g_verbose) {
        fprintf(stderr, "%s", text);
    }
}

static void null_log_callback(ggml_log_level level, const char* text, void* user_data) {
    // Completely suppress llama.cpp logging
}

extern "C" {

JNIEXPORT void JNICALL Java_fastaimodel_FastAIModel_nativeSetVerbose(JNIEnv* env, jclass clazz, jboolean verbose) {
    g_verbose = (verbose == JNI_TRUE);
    if (g_verbose) {
        llama_log_set(nullptr, nullptr);
    } else {
        llama_log_set(null_log_callback, nullptr);
    }
}

JNIEXPORT jlong JNICALL Java_fastaimodel_FastAIModel_nativeInit(
    JNIEnv* env, jclass clazz, jstring jModelPath, jint ctxSize, jint gpuLayers) {

    if (!g_verbose) {
        llama_log_set(null_log_callback, nullptr);
    } else {
        llama_log_set(nullptr, nullptr);
    }

#ifdef _WIN32
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
    AddDllDirectory(L"lib");
    AddDllDirectory(L".");

    char dllPath[MAX_PATH] = {0};
    HMODULE hModule = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCSTR)&Java_fastaimodel_FastAIModel_nativeInit, &hModule)) {
        GetModuleFileNameA(hModule, dllPath, MAX_PATH);
        std::string sPath(dllPath);
        size_t lastSep = sPath.find_last_of("\\/");
        if (lastSep != std::string::npos) {
            std::string dir = sPath.substr(0, lastSep);
            std::string baseDll = dir + "\\ggml-base.dll";
            std::string ompDll  = dir + "\\libomp140.x86_64.dll";

            LoadLibraryA(ompDll.c_str());
            LoadLibraryA(baseDll.c_str());

            // Load fastest available CPU backend: AVX2 > icelake > haswell > x64
            const char* cpuVariants[] = {
                "ggml-cpu-avx2.dll",
                "ggml-cpu-icelake.dll",
                "ggml-cpu-haswell.dll",
                "ggml-cpu-x64.dll"
            };
            bool loaded = false;
            for (const char* variant : cpuVariants) {
                std::string dll = dir + "\\" + variant;
                if (ggml_backend_load(dll.c_str()) != nullptr) {
                    if (g_verbose) fprintf(stderr, "[FastAIModel] CPU backend: %s\n", variant);
                    loaded = true;
                    break;
                }
            }
            ggml_backend_load_all_from_path(dir.c_str());
        }
    }
#endif

    llama_backend_init();
    ggml_backend_load_all();

    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpuLayers;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    llama_model* model = llama_load_model_from_file(modelPath, mparams);

    if (!model) {
        printf("[FastAIModel C++] Error: llama_load_model_from_file failed for path: %s\n", modelPath);
        fflush(stdout);
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        return 0;
    }
    env->ReleaseStringUTFChars(jModelPath, modelPath);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = ctxSize;
    cparams.n_batch  = 2048;
    cparams.n_ubatch = 2048;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k   = GGML_TYPE_Q4_0;
    cparams.type_v   = GGML_TYPE_Q4_0;
    cparams.offload_kqv = true;
    cparams.op_offload  = true;
    int threadCount = (int)std::thread::hardware_concurrency();
    cparams.n_threads = threadCount;
    cparams.n_threads_batch = threadCount;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        return 0;
    }

    auto* handle = new FastAIModelHandle();
    handle->model = model;
    handle->ctx = ctx;
    handle->vocab = llama_model_get_vocab(model);

    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL Java_fastaimodel_FastAIModel_nativePredict(
    JNIEnv* env, jclass clazz, jlong handlePtr, jstring jPrompt, jint maxTokens,
    jfloat temperature, jfloat topP, jobject callback) {

    auto* handle = reinterpret_cast<FastAIModelHandle*>(handlePtr);
    if (!handle) return;

    if (!callback) return;
    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    if (!onTokenMethod) return;

    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);

    // Tokenize
    int32_t n_tokens_max = -llama_tokenize(handle->vocab, prompt, (int32_t)strlen(prompt), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens_max);
    int32_t n_tokens = llama_tokenize(handle->vocab, prompt, (int32_t)strlen(prompt), tokens.data(), tokens.size(), true, true);
    env->ReleaseStringUTFChars(jPrompt, prompt);

    if (n_tokens < 0) return;
    tokens.resize(n_tokens);

    // Decode prompt in chunks of 512 tokens using llama_batch_get_one
    size_t chunk_size = 512;
    for (size_t i = 0; i < tokens.size(); i += chunk_size) {
        size_t n_eval = std::min(chunk_size, tokens.size() - i);
        struct llama_batch batch = llama_batch_get_one(tokens.data() + i, (int32_t)n_eval);
        if (llama_decode(handle->ctx, batch) != 0) {
            return;
        }
    }

    // Sampler setup
    struct llama_sampler* smpl = nullptr;
    if (temperature <= 0.0f) {
        smpl = llama_sampler_init_greedy();
    } else {
        struct llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(1234));
        smpl = chain;
    }

    // Prediction loop
    for (int i = 0; i < maxTokens; i++) {
        llama_token token = llama_sampler_sample(smpl, handle->ctx, -1);

        if (token == llama_vocab_eos(handle->vocab)) {
            break;
        }

        // Decode token to string piece
        char buf[256];
        int32_t len = llama_token_to_piece(handle->vocab, token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            std::string piece(buf, len);
            jstring jPiece = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, onTokenMethod, jPiece);
            env->DeleteLocalRef(jPiece);
        }

        // Decode next token (automatically tracks position since pos is NULL)
        struct llama_batch next_batch = llama_batch_get_one(&token, 1);
        if (llama_decode(handle->ctx, next_batch) != 0) {
            break;
        }
    }

    if (smpl) {
        llama_sampler_free(smpl);
    }
}

JNIEXPORT void JNICALL Java_fastaimodel_FastAIModel_nativeFree(
    JNIEnv* env, jclass clazz, jlong handlePtr) {

    auto* handle = reinterpret_cast<FastAIModelHandle*>(handlePtr);
    if (!handle) return;

    if (handle->ctx) {
        llama_free(handle->ctx);
    }
    if (handle->model) {
        llama_free_model(handle->model);
    }
    delete handle;
    
    llama_backend_free();
}

}
