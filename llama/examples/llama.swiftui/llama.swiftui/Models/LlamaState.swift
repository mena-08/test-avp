import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(
                    name: "llama-2-7b (Q2_K, 1.6 GiB)",
                    url: "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf?download=true",
                    filename: "llama-2-7b-chat.Q2_K.gguf", status: "download"
                ),

         Model(
                    name: "llama-2-7b (Q3_K_S, 1.6 GiB)",
                    url: "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q3_K_S.gguf?download=true",
                    filename: "llama-2-7b-chat.Q3_K_S.gguf", status: "download"
                ),


         Model(
                    name: "qwen2-0_5b (fp16, 1.6 GiB)",
                    url: "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-fp16.gguf?download=true",
                    filename: "qwen2-0_5b-instruct-fp16.gguf", status: "download"
                ),

         Model(
                    name: "Mistral-7B (IQ1_M, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ1_M.gguf?download=true",
                    filename: "Mistral-7B-Instruct-v0.3-IQ1_M.gguf", status: "download"
                ),

         Model(
                    name: "Mistral-7B (IQ2_XS, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ2_XS.gguf?download=true",
                    filename: "Mistral-7B-Instruct-v0.3-IQ2_XS.gguf", status: "download"
                ),

         Model(
                    name: "Mistral-7B (IQ3_XS, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ3_XS.gguf?download=true",
                    filename: "Mistral-7B-Instruct-v0.3-IQ3_XS.gguf", status: "download"
                ),

         Model(
                    name: "Mistral-7B (IQ4_XS, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-IQ4_XS.gguf?download=true",
                    filename: "Mistral-7B-Instruct-v0.3-IQ4_XS.gguf", status: "download"
                ),

         Model(
                    name: "Vikhr-Gemma-2B (Q3_K_M, 1.6 GiB)",
                    url: "https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct-GGUF/resolve/main/Vikhr-Gemma-2B-instruct-Q3_K_M.gguf?download=true",
                    filename: "Vikhr-Gemma-2B-instruct-Q3_K_M.gguf", status: "download"
                ),
         Model(
                    name: "Vikhr-Gemma-2B (Q4_0, 1.6 GiB)",
                    url: "https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct-GGUF/resolve/main/Vikhr-Gemma-2B-instruct-Q4_0.gguf?download=true",
                    filename: "Vikhr-Gemma-2B-instruct-Q4_0.gguf", status: "download"
                ),
         Model(
                    name: "Vikhr-Gemma-2B (Q5_0, 1.6 GiB)",
                    url: "https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct-GGUF/resolve/main/Vikhr-Gemma-2B-instruct-Q5_0.gguf?download=true",
                    filename: "Vikhr-Gemma-2B-instruct-Q5_0.gguf", status: "download"
                ),
         Model(
                    name: "Vikhr-Gemma-2B (Q6_K, 1.6 GiB)",
                    url: "https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct-GGUF/resolve/main/Vikhr-Gemma-2B-instruct-Q6_K.gguf?download=true",
                    filename: "Vikhr-Gemma-2B-instruct-Q6_K.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q2_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q2_K.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q2_K.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q3_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q3_K_L.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q3_K_L.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q4_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q4_K_L.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q4_K_L.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q5_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q5_K_L.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q5_K_L.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q6_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q6_K.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q6_K.gguf", status: "download"
                ),
         Model(
                    name: "Phi-3.1-mini-4k (Q8_0, 1.6 GiB)",
                    url: "https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF/resolve/main/Phi-3.1-mini-4k-instruct-Q8_0.gguf?download=true",
                    filename: "Phi-3.1-mini-4k-instruct-Q8_0.gguf", status: "download"
                )
    ]
    func loadModel(modelUrl: URL?) throws {
        if let modelUrl {
            messageLog += "Loading model...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(await llamaContext.n_len) / t_generation

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }
        
        // Define the thread counts you want to test
        //let threadCounts = [1, 2, 4, 8, 16, 32]
        
        //LOXR-TEST
        //As we discussed, we will use 4 threads for all tests
        //we override the amount of threads before any test
        let threadCounts = [4]

        messageLog += "\nRunning benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        // Perform the benchmark for each thread count
        for threadCount in threadCounts {
            messageLog += "\nSetting threads to \(threadCount)...\n"
            await llamaContext.change_threads(n_threads: Int32(threadCount))

            let t_start = DispatchTime.now().uptimeNanoseconds
            let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
            let t_end = DispatchTime.now().uptimeNanoseconds

            let t_heat = Double(t_end - t_start) / NS_PER_S
            messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

            // If heat up takes too long, abort
            //LOXR-TEST
            //FORCE BENCHMARK
            if t_heat > 1.0 {
                messageLog += "Heat up time is too long, aborting benchmark for threads \(threadCount)\n"
                continue
            }

            //LOXR-TEST
            //These values should be modified according which tests do we want to perform
            let ppValues = [1,16,32,64,256,512]
            let tgValues = [1,16,32,64,256,512]
            let fixedPl = 1

            //LOXR-TEST
            // Loop to run the benchmark 5 or 20, you choose times
            for iteration in 1...5 {
                print("Starting iteration \(iteration)")

                // Loop through combinations of pp and tg values
                for pp in ppValues {
                    for tg in tgValues {
                        let partialResults = "\nRunning benchmark with \(threadCount) threads, pp: \(pp), tg: \(tg), pl: \(fixedPl)\n"

                        messageLog += partialResults
                        let result = await llamaContext.bench(pp: pp, tg: tg, pl: fixedPl, nr: 5)

                        // Append the result after running the benchmark
                        let benchmarkResult = "Benchmark result for iteration \(iteration), threads: \(threadCount), pp: \(pp), tg: \(tg), pl: \(fixedPl):\n\(result)\n"
                        print(benchmarkResult)

                        // Optionally add the result to the messageLog
                        messageLog += benchmarkResult
                    }
                }
                
                print("Completed iteration \(iteration)")
            }

        }

        messageLog += "\nBenchmark complete.\n"
    }


    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }
}
