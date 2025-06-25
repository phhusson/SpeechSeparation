#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <fftw3.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <ladspa.h>

constexpr int chunk_size = 1024;
constexpr int fft_size = 2048;
constexpr int fft_n_entries = 1 + fft_size / 2; // Symetric + DC
constexpr int fft_n_entries_sym = fft_size / 2 - 1;
constexpr int n_chunks = fft_size / chunk_size;
constexpr int n_channels = 2;

// LADSPA plugin
class SpeechSeparator {
public:

    LADSPA_Data *control = nullptr;
    LADSPA_Data *input1 = nullptr;
    LADSPA_Data *input2 = nullptr;
    LADSPA_Data *output1 = nullptr;
    LADSPA_Data *output2 = nullptr;

    Ort::Env ortEnv = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::Session *ortSession;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    fftw_complex *fft_time = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * fft_size));
    fftw_complex *fft_freq = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * fft_size));
    fftw_plan fft_plan_fwd = fftw_plan_dft_1d(fft_size, fft_time, fft_freq, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan fft_plan_inv = fftw_plan_dft_1d(fft_size, fft_freq, fft_time, FFTW_BACKWARD, FFTW_ESTIMATE);
    float fft_window[fft_size]{};

    float tensor_in[n_channels * fft_n_entries * 2] = {};
    int64_t shape_data[2] = {n_channels, fft_n_entries*2};
    float *state_in;
    int64_t state_size;
    Ort::Value *inputs;

    float tensor_out[n_channels * fft_n_entries * 2] = {};
    float *state_out;
    Ort::Value *outputs;

    // Store the result of the last 8 ifft to be able overlap and add them
    float overlap[n_chunks][fft_size] = {};
    // Used for the 512-hop 4096 fft
    float current_chunk[n_channels * fft_size] = {};

    int ladspa_pos = 0;
    float ladspa_buffer_in[n_channels][chunk_size] = {};
    float ladspa_buffer_out[n_channels][chunk_size] = {};

    SpeechSeparator() {
        // Initialize fft_window with a Hann window
        for (int i = 0; i < fft_size; i++) {
            fft_window[i] = 0.5 * (1 - cos(2 * M_PI * i / fft_size));
        }

std::cerr << "Constants:" << std::endl;
std::cerr << "chunk_size: " << chunk_size << std::endl;
std::cerr << "fft_size: " << fft_size << std::endl;
std::cerr << "fft_n_entries: " << fft_n_entries << std::endl;
std::cerr << "fft_n_entries_sym: " << fft_n_entries_sym << std::endl;
std::cerr << "n_chunks: " << n_chunks << std::endl;
std::cerr << "n_channels: " << n_channels << std::endl;


        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.AddConfigEntry("session.allow_spinning", "0");
        ortSession = new Ort::Session(ortEnv, "/home/phh/CLionProjects/SpeechRecognition-onnx/model.onnx", options);

        state_in = state_out = nullptr;
        state_size = 0;
        outputs = inputs = nullptr;

        int nInputs = ortSession->GetInputCount();
        std::cerr << "Inputs:" << std::endl;
        Ort::AllocatorWithDefaultOptions allocator;
        for(int i = 0; i < nInputs; i++) {
            auto name = ortSession->GetInputNameAllocated(i, allocator);
            auto typeInfo = ortSession->GetInputTypeInfo(i);
            ONNXType t = typeInfo.GetONNXType();

            std::cerr << name.get() << std::endl;
            if (t != ONNX_TYPE_TENSOR) continue;

            if (strcmp(name.get(), "state.0") == 0) {
                auto typeInfo2 = typeInfo.GetTensorTypeAndShapeInfo();
                auto shape = typeInfo2.GetShape();
                int product = 1;
                
                int64_t *state_shape = new int64_t[shape.size()]();
                for(int j=0; j<shape.size(); j++) {
                    int64_t a = shape[j];
                    state_shape[j] = a;
                    product *= a;
                }
                state_size = product;
                state_in = new float[state_size]();
                state_out = new float[state_size]();

                inputs = new Ort::Value[] {
                    Ort::Value::CreateTensor<float>(memory_info, tensor_in, n_channels * fft_n_entries * 2, shape_data, 2),
                    Ort::Value::CreateTensor<float>(memory_info, state_in, product, state_shape, shape.size())
                };
                outputs = new Ort::Value[] {
                    Ort::Value::CreateTensor<float>(memory_info, tensor_out, n_channels * fft_n_entries * 2, shape_data, 2),
                    Ort::Value::CreateTensor<float>(memory_info, state_out, product, state_shape, shape.size())
                };
std::cerr << "Done checking size of state.0 with " << state_size << std::endl;
                break;
            }
        }
        if (state_in == nullptr) {
            std::cerr << "Failed checking state infos" << std::endl;
        }
    }

    virtual ~SpeechSeparator() {
std::cerr << "Called destructor" << std::endl;
        delete ortSession;
        delete[] inputs;
        delete[] outputs;
        delete[] state_in;
        delete[] state_out;
    }

    void connect(unsigned long port, LADSPA_Data *data) {
        switch (port) {
            case 0:
                control = data;
                break;
            case 1:
                input1 = data;
                break;
            case 2:
                input2 = data;
                break;
            case 3:
                output1 = data;
                break;
            case 4:
                output2 = data;
                break;
            default: break;
        }
    }

    void run(unsigned long sampleCount) {
        int pos = 0;
        while (sampleCount > 0) {
            int to_copy = std::min(chunk_size - ladspa_pos, (int)sampleCount);
            for (int i = 0; i < to_copy; i++) {
                ladspa_buffer_in[0][ladspa_pos] = input1[pos + i];
                ladspa_buffer_in[1][ladspa_pos] = input2[pos + i];
                output1[pos + i] = ladspa_buffer_out[0][ladspa_pos];
                output2[pos + i] = ladspa_buffer_out[1][ladspa_pos];
                ladspa_pos++;
            }
            sampleCount -= to_copy;
            pos += to_copy;
            if (ladspa_pos == chunk_size) {
                new_chunk();
            }
        }
    }

    void new_chunk() {
        if (ladspa_pos != chunk_size) {
            fprintf(stderr, "Invalid state: ladspa_pos = %d\n", ladspa_pos);
            return;
        }

        // Move the last (4096-512) samples to the beginning of the buffer
        for(int j = 0; j < n_channels; j++) {
            int channel_offset = j * fft_size;
            memmove(current_chunk + channel_offset, current_chunk + channel_offset + chunk_size, sizeof(float) * (fft_size - chunk_size));
        }

        for (int j = 0; j < n_channels; j++) {
            int channel_offset = j * fft_size;
            for (int i = 0; i < chunk_size; i++) {
                current_chunk[ channel_offset + fft_size - chunk_size + i] = ladspa_buffer_in[0][i];
            }
        }

        // Copy the left chunks to the fft input buffer, with the window
        for (int i = 0; i < fft_size; i++) {
            fft_time[i][0] = current_chunk[i] * fft_window[i];
            fft_time[i][1] = 0;
        }
        // Do the fft
        fftw_execute(fft_plan_fwd);

        // Copy the result of the fft into tensor_in, ignoring the leading 2047 values which are symetric since it's a real input
        for (int j =0; j < n_channels; j++) {
            int channel_offset = 2 * fft_n_entries * j;
            for (int i = 0; i < fft_n_entries; i++) {
                tensor_in[channel_offset * j + i * 2] = fft_freq[i][0];
                tensor_in[channel_offset * j + i * 2 + 1] = fft_freq[i][1];
            }
        }

        // Infer the model
        Ort::RunOptions run_options;
        const char* input_names[] = {"x.0", "state.0"};
        const char* output_names[] = {"y.0", "new_state.0"};
        ortSession->Run(run_options, input_names, inputs, 2, output_names, outputs, 2);

        // Copy the output of the model to the fft input buffer
        // Also apply the requested mix
        float mix = *control;
        if (mix >= 0.0) {
            for (int i = 0; i < fft_n_entries; i++) {
                fft_freq[i][0] = mix * tensor_out[i * 2] + (1.0 - mix) * fft_freq[i][0];
                fft_freq[i][1] = mix * tensor_out[i * 2 + 1] + (1.0 - mix) * fft_freq[i][1];
            }
        } else {
            for (int i = 0; i < fft_n_entries; i++) {
                fft_freq[i][0] = mix * tensor_out[i * 2] + fft_freq[i][0];
                fft_freq[i][1] = mix * tensor_out[i * 2 + 1] +  fft_freq[i][1];
            }
        }

        // And completment the symetric part
        for (int i = 0; i < fft_n_entries_sym; i++) {
            // We want 2050 to be like 2048, 2051 to be like 2047, etc
            fft_freq[fft_n_entries + i][0] = fft_freq[fft_n_entries_sym - i][0];
            fft_freq[fft_n_entries + i][1] = -fft_freq[fft_n_entries_sym - i][1];
        }

        // Do the inverse fft
        fftw_execute(fft_plan_inv);

        // Move overlaps by one to the left
        for (int i = 0; i < n_chunks-1; i++) {
            memcpy(overlap[i], overlap[i+1], fft_size * sizeof(float));
        }

        // Copy the result of the ifft to the overlap
        for (int i = 0; i < fft_size; i++) {
            overlap[n_chunks-1][i] = fft_time[i][0] / fft_size;
        }

        // Now look the to the first 512 samples we just added (matching samples in all overlaps), and merge them using the hann window
        for (int i = 0; i < chunk_size; i++) {
            float sum = 0;
            float window_sum = 0;
            for (int j = 0; j < n_chunks; j++) {
                int pos = i + (n_chunks - 1 - j) * chunk_size;
                sum += overlap[j][pos];
                window_sum += fft_window[pos];
            }
            float s = sum / window_sum;
            for (int j = 0; j<n_channels; j++) {
                ladspa_buffer_out[j][i] = s;
            }
        }

        // Copy state to state_in
        memcpy(state_in, state_out, state_size*sizeof(float));
        //memset(state_in, 0, state_size * sizeof(float));
        ladspa_pos = 0;
    }

};

static LADSPA_Handle instantiate(const LADSPA_Descriptor*descriptor, unsigned long sampleRate) {
    auto separator = new SpeechSeparator();
    return separator;
}

static void connect_port(LADSPA_Handle instance, unsigned long port, LADSPA_Data *data) {
    auto separator = reinterpret_cast<SpeechSeparator*>(instance);
    separator->connect(port, data);
}

static void run(LADSPA_Handle instance, unsigned long sampleCount) {
    auto separator = reinterpret_cast<SpeechSeparator*>(instance);
    separator->run(sampleCount);
}

static void cleanup(LADSPA_Handle instance) {
    auto separator = reinterpret_cast<SpeechSeparator*>(instance);
    delete separator;
}

LADSPA_Descriptor *g_descriptor;

static void __attribute__((constructor)) init() {
    g_descriptor = new LADSPA_Descriptor();
    g_descriptor->UniqueID = 0xf433b044;
    g_descriptor->Label = strdup("speech_separator");
    g_descriptor->Properties = 0;
    g_descriptor->Name = strdup("Speech Separator");
    g_descriptor->Maker = strdup("Pierre-Hugues Husson @ Freebox");
    g_descriptor->Copyright = strdup("None");
    g_descriptor->PortCount = 5;

    auto portDescriptors = new LADSPA_PortDescriptor[5];
    portDescriptors[0] = LADSPA_PORT_INPUT | LADSPA_PORT_CONTROL;
    portDescriptors[1] = LADSPA_PORT_INPUT | LADSPA_PORT_AUDIO;
    portDescriptors[2] = LADSPA_PORT_INPUT | LADSPA_PORT_AUDIO;
    portDescriptors[3] = LADSPA_PORT_OUTPUT | LADSPA_PORT_AUDIO;
    portDescriptors[4] = LADSPA_PORT_OUTPUT | LADSPA_PORT_AUDIO;
    g_descriptor->PortDescriptors = portDescriptors;

    auto portNames = new char*[5];
    portNames[0] = strdup("Control");
    portNames[1] = strdup("Input (Left)");
    portNames[2] = strdup("Input (Right)");
    portNames[3] = strdup("Output (Left)");
    portNames[4] = strdup("Output (Right)");
    g_descriptor->PortNames = portNames;

    auto portRangeHints = new LADSPA_PortRangeHint[5];
    portRangeHints[0].HintDescriptor = LADSPA_HINT_DEFAULT_1 | LADSPA_HINT_BOUNDED_BELOW;
    portRangeHints[0].LowerBound = 0;
    portRangeHints[1].HintDescriptor = 0;
    portRangeHints[2].HintDescriptor = 0;
    portRangeHints[3].HintDescriptor = 0;
    portRangeHints[4].HintDescriptor = 0;

    g_descriptor->PortRangeHints = portRangeHints;

    // TODO: port range hints?
    g_descriptor->instantiate = instantiate;
    g_descriptor->connect_port = connect_port;
    g_descriptor->activate = NULL;
    g_descriptor->run = run;
    g_descriptor->run_adding = NULL;
    g_descriptor->deactivate = NULL;
    g_descriptor->cleanup = cleanup;
}

extern "C" const LADSPA_Descriptor* ladspa_descriptor(unsigned long idx) {
    if (idx == 0) return g_descriptor;
    return NULL;
}
