// req_generator.cpp
#include "compass/request_generator.h"
#include "compass/utils_compass.h"

using json = nlohmann::json;

Req::Req(int req_id, Req::Type req_type, int req_lens, int req_his_lens)
    : id(req_id), type(req_type), lens(req_lens), his_lens(req_his_lens) {}

std::string ReqGenerator::inputLengthsFile="./config/simulated_input_lengths.json";
std::string ReqGenerator::outputLengthsFile="./config/simulated_output_lengths.json";

ReqGenerator::ReqGenerator(int batch_size) 
    : now_id(0),
      batch_size(batch_size),
      warmup(5000){
    
    req_cache.resize(batch_size, {0, std::nullopt});

    // Load JSON data
    std::ifstream input_file(inputLengthsFile);
    assert(input_file);
    json input_json;
    input_file >> input_json;
    simulated_input_lengths = input_json.get<std::vector<int>>();
    input_index = 0;

    std::ifstream output_file(outputLengthsFile);
    assert(output_file);
    json output_json;
    output_file >> output_json;
    simulated_output_lengths = output_json.get<std::vector<int>>();
    output_index = 0;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        warmupGenerate();
    }
}

void ReqGenerator::warmupGenerate() {
    for (int i = 0; i < batch_size; ++i) {
        auto& [seq_len, req_opt] = req_cache[i];
        if (seq_len == 0) {
            int input_len = getNextInputLength();
            int output_len = getNextOutputLength();
            
            req_opt = Req(now_id, Req::Type::Prefill, input_len, 0);
            seq_len = output_len - 1;
            now_id++;
        } else {
            if (req_opt.has_value()) {
                Req& req = req_opt.value();
                req.type = Req::Type::Decode;
                req.lens = 1;
                req.his_lens += 1;
                seq_len--;
            }
        }
    }
}

int ReqGenerator::getNextInputLength() {
    return std::max(1, simulated_input_lengths[input_index++]);
}

int ReqGenerator::getNextOutputLength() {
    return std::max(1, simulated_output_lengths[output_index++]);
}

batchedReqs_t ReqGenerator::generateReq(int micro_batch_size) {
    assert(batch_size % micro_batch_size == 0);
    batchedReqs_t result;

    if (res_reqs.empty()) {
        res_reqs.clear();
        for (int i = 0; i < batch_size; ++i) {
            auto& [seq_len, req_opt] = req_cache[i];
            if (seq_len == 0) {
                int input_len = getNextInputLength();
                int output_len = getNextOutputLength();
                
                req_opt = Req(now_id, Req::Type::Prefill, input_len, 0);
                seq_len = output_len - 1;
                res_reqs.push_back(*req_opt);
                now_id++;
            } else {
                if (req_opt.has_value()) {
                    Req& req = req_opt.value();
                    req.type = Req::Type::Decode;
                    req.lens = 1;
                    req.his_lens += 1;
                    seq_len--;
                    res_reqs.push_back(req);
                }
            }
        }

        for (int i = 0; i < batch_size; ++i) {
            if (ThreadSafeRandom::rand_percent() < 0.3) {
                // int input_len = getNextInputLength();
                int input_len = 32;
                int output_len = getNextOutputLength();
                
                Req new_req(now_id, Req::Type::Prefill, input_len, 0);
                req_cache[i] = {output_len - 1, new_req};
                res_reqs[i] = new_req;
                now_id++;
            }
        }
    }

    for (int i = 0; i < batch_size / micro_batch_size; ++i) {
        auto start = res_reqs.begin() + i * micro_batch_size;
        auto end = start + micro_batch_size;
        result.emplace_back(start, end);
    }

    return result;
}