#include <iostream>
#include <torch/script.h>
#include <fftw3.h>
#include <unistd.h>
#include <cmath>
#include <ladspa.h>

constexpr int chunk_size = 1024;
constexpr int n_overlaps = 4096 / chunk_size;
constexpr int band_features = 32;
// LADSPA plugin
class SpeechSeparator {
public:


    LADSPA_Data *control = nullptr;
    LADSPA_Data *input1 = nullptr;
    LADSPA_Data *input2 = nullptr;
    LADSPA_Data *output1 = nullptr;
    LADSPA_Data *output2 = nullptr;

    torch::jit::Module model1,model2;

    fftw_complex *fft_time = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * 4096));
    fftw_complex *fft_freq = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * 4096));
    fftw_plan fft_plan_fwd = fftw_plan_dft_1d(4096, fft_time, fft_freq, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan fft_plan_inv = fftw_plan_dft_1d(4096, fft_freq, fft_time, FFTW_BACKWARD, FFTW_ESTIMATE);
    float fft_window[4096]{};

    at::TensorOptions floatTensorOnCpuType = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);

    float tensor_in[2 * 2049 * 2] = {};
    int64_t shape_in[2] = {2, 2049*2};
    float state_in[4 * 2 * 40 * band_features] = {};
    int64_t state_shape[5] = {4, 2, 40, band_features};
    torch::jit::IValue tensor_in_v = at::from_blob(tensor_in, {2, 4098}, floatTensorOnCpuType);
    torch::jit::IValue state_in_v = at::from_blob(state_in, {4, 2, 40, band_features}, floatTensorOnCpuType);

    // Store the result of the last 8 ifft to be able overlap and add them
    float overlap[8][4096] = {};
    // Used for the 512-hop 4096 fft
    float current_chunk[2 * 4096] = {};

    int ladspa_pos = 0;
    float ladspa_buffer_in[2][chunk_size] = {};
    float ladspa_buffer_out[2][chunk_size] = {};

    SpeechSeparator() {
        // Initialize fft_window with a Hann window
        for (int i = 0; i < 4096; i++) {
            fft_window[i] = 0.5 * (1 - cos(2 * M_PI * i / 4096));
        }
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
        model1 = torch::jit::load("model-script-1.pt");
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
        model1.to(at::kCPU);
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
        model1.eval();
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);

        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
        model2 = torch::jit::load("model-script-2.pt");
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
        model2.eval();
        fprintf(stderr, "%s %d\n", __FUNCTION__, __LINE__);
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
        for (int i = 0; i < 4096 - chunk_size; i++) {
            current_chunk[i] = current_chunk[i + chunk_size];
            current_chunk[i + 4096] = current_chunk[i + 4096 + chunk_size];
        }

        for (int i = 0; i < chunk_size; i++) {
            current_chunk[4096 - chunk_size + i] = ladspa_buffer_in[0][i];
            current_chunk[4096 + 4096 - chunk_size + i] = ladspa_buffer_in[0][i];
        }

                // Copy the left chunks to the fft input buffer, with the window
        for (int i = 0; i < 4096; i++) {
            fft_time[i][0] = current_chunk[i] * fft_window[i];
            fft_time[i][1] = 0;
        }
        // Do the fft
        fftw_execute(fft_plan_fwd);

        // Copy the result of the fft into tensor_in, ignoring the leading 2047 values which are symetric since it's a real input
        for (int i = 0; i < 2049; i++) {
            tensor_in[i * 2] = fft_freq[i][0];
            tensor_in[i * 2 + 1] = fft_freq[i][1];
            tensor_in[i * 2 + 2049 * 2] = fft_freq[i][0];
            tensor_in[i * 2 + 2049 * 2 + 1] = fft_freq[i][1];
        }

        // Infer the model
        //auto ret = model.forward({inputs[0].toTensor().to(at::kVulkan), inputs[1].toTensor().to(at::kVulkan)});
        auto band_outputs = model1.forward({tensor_in_v}).toTensor();
        auto ret = model2.forward({tensor_in_v.toTensor(), band_outputs, state_in_v.toTensor()});
        const auto& retTuple = ret.toTuple()->elements();
        auto output_x_t = retTuple[0].toTensor();
        float *output_x = output_x_t.data_ptr<float>();
        auto output_state_t = retTuple[1].toTensor();
        float *state_out = output_state_t.data_ptr<float>();

        // Copy the output of the model to the fft input buffer
        // Also apply the requested mix
        float mix = *control;
        for (int i = 0; i < 2049; i++) {
            fft_freq[i][0] = mix * output_x[2*i] + (1.0 - mix) * fft_freq[i][0];
            fft_freq[i][1] = mix * output_x[2*i+1] + (1.0 - mix) * fft_freq[i][1];
        }

        // And completment the symetric part
        for (int i = 0; i < 2047; i++) {
            // We want 2050 to be like 2048, 2051 to be like 2047, etc
            fft_freq[2049 + i][0] = fft_freq[2047 - i][0];
            fft_freq[2049 + i][1] = -fft_freq[2047 - i][1];
        }

        // Do the inverse fft
        fftw_execute(fft_plan_inv);

        // Move overlaps by one to the left
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 4096; j++) {
                overlap[i][j] = overlap[i + 1][j];
            }
        }

        // Copy the result of the ifft to the overlap
        for (int i = 0; i < 4096; i++) {
            overlap[7][i] = fft_time[i][0] / 4096;
        }

        // Now look the to the first 512 samples we just added (matching samples in all overlaps), and merge them using the hann window
        for (int i = 0; i < chunk_size; i++) {
            float sum = 0;
            float window_sum = 0;
            for (int j = 0; j < n_overlaps; j++) {
                int pos = i + (n_overlaps - 1 - j) * chunk_size;
                sum += overlap[j][pos];
                window_sum += fft_window[pos];
            }
            float s = sum / window_sum;
            ladspa_buffer_out[0][i] = s;
            ladspa_buffer_out[1][i] = s;
        }

        // Copy state to state_in
        memcpy(state_in, state_out, sizeof(state_in));
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
