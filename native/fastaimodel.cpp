#include <jni.h>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "llama.h"

struct FastAIModelHandle {
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;
};

extern "C" {

JNIEXPORT jlong JNICALL Java_fastaimodel_FastAIModel_nativeInit(
    JNIEnv* env, jclass clazz, jstring jModelPath, jint ctxSize, jint gpuLayers) {

    llama_backend_init();

    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpuLayers;

    llama_model* model = llama_load_model_from_file(modelPath, mparams);
    env->ReleaseStringUTFChars(jModelPath, modelPath);

    if (!model) return 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = ctxSize;

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

    // Decode prompt
    struct llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(handle->ctx, batch) != 0) {
        return;
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
