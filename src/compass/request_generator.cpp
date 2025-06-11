// req_generator.cpp
#include "compass/request_generator.h"
#include "compass/utils_compass.h"
#include "debug.h"

using json = nlohmann::json;

Req::Req(int req_id, Req::Type req_type, int req_lens, int req_his_lens)
    : id(req_id), type(req_type), lens(req_lens), his_lens(req_his_lens) {}

std::string ReqGenerator::inputLengthsFile="./config/simulated_input_lengths.json";
std::string ReqGenerator::outputLengthsFile="./config/simulated_output_lengths.json";

ReqGenerator::ReqGenerator(int batch_size) 
    : now_id(0),
      batch_size(batch_size),
      warmup(4500){
    
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

    std::vector<Req> prefill_reqs;
    std::vector<Req> decode_reqs;

    for (int i = 0; i < batch_size; ++i) {
        auto& [seq_len, req_opt] = req_cache[i];

        if (seq_len == 0) {
            int input_len = getNextInputLength();
            int output_len = getNextOutputLength();
            Req new_req(now_id, Req::Type::Prefill, input_len, 0);
            req_cache[i] = {output_len - 1, new_req};
            prefill_reqs.push_back(new_req);
            now_id++;
        } else {
            if (req_opt.has_value()) {
                Req& req = req_opt.value();
                req.type = Req::Type::Decode;
                req.lens = 1;
                req.his_lens += 1;
                seq_len--;
                decode_reqs.push_back(req);
            }
        }
    }

    // 将 prefill 放在 decode 前面
    res_reqs.clear();
    res_reqs.insert(res_reqs.end(), prefill_reqs.begin(), prefill_reqs.end());
    res_reqs.insert(res_reqs.end(), decode_reqs.begin(), decode_reqs.end());

    // 拆 micro-batch
    batchedReqs_t result;
    for (int i = 0; i < batch_size / micro_batch_size; ++i) {
        auto start = res_reqs.begin() + i * micro_batch_size;
        auto end = start + micro_batch_size;
        result.emplace_back(start, end);
    }

    return result;
}

batchedReqs_t ReqGenerator::generateReq(int micro_batch_size, int num_prefill, int num_decode) {
    assert(batch_size % micro_batch_size == 0);
    assert(num_prefill + num_decode == batch_size);

    batchedReqs_t result;
    res_reqs.clear();
    std::vector<bool> used(batch_size, false);
    int prefill_count = 0;
    int decode_count = 0;

    std::vector<Req> prefill_reqs;
    std::vector<Req> decode_reqs;

    // Step 1: 收集 prefill 请求（不写入 req_cache）
    for (int i = 0; i < batch_size && prefill_count < num_prefill; ++i) {
        int input_len = getNextInputLength();
        Req new_req(now_id, Req::Type::Prefill, input_len, 0);
        prefill_reqs.push_back(new_req);
        now_id++;
        prefill_count++;
        // 不写回 req_cache，避免影响自然分布
    }

    // Step 2: 从 req_cache 中找合法的 decode 请求
    for (int i = 0; i < batch_size && decode_count < num_decode; ++i) {
        auto& [seq_len, req_opt] = req_cache[i];
        if (used[i] || !req_opt.has_value()) continue;
        if (seq_len > 0) {
            Req& req = req_opt.value();
            req.type = Req::Type::Decode;
            req.lens = 1;
            req.his_lens += 1;
            seq_len--;
            decode_reqs.push_back(req);
            used[i] = true;
            decode_count++;
        }
        if (seq_len == 0) {
            int input_len = getNextInputLength();
            int output_len = getNextOutputLength();
            Req new_req(now_id, Req::Type::Prefill, input_len, 0);
            req_cache[i] = {output_len - 1, new_req};
            now_id++;
        }
    }

    // Step 3: decode fallback — 用合法 prefill 转成 decode
    for (int i = 0; i < batch_size && decode_count < num_decode; ++i) {
        auto& [seq_len, req_opt] = req_cache[i];
        if (used[i] || !req_opt.has_value()) continue;
        Req& req = req_opt.value();
        if (req.type == Req::Type::Prefill && req.his_lens == 0 && seq_len > 0) {
            req.type = Req::Type::Decode;
            req.lens = 1;
            req.his_lens += 1;
            seq_len--;
            decode_reqs.push_back(req);
            used[i] = true;
            decode_count++;
        }
        if (seq_len == 0) {
            int input_len = getNextInputLength();
            int output_len = getNextOutputLength();
            Req new_req(now_id, Req::Type::Prefill, input_len, 0);
            req_cache[i] = {output_len - 1, new_req};
            now_id++;
        }
    }

    // Step 4: 严格验证是否满足需求
    assert(prefill_count == num_prefill && "Prefill count mismatch!");
    assert(decode_count == num_decode && "Not enough valid decode requests!");

    auto sort_by_lens_desc = [](const Req& a, const Req& b) {
        if (a.lens != b.lens)
            return a.lens > b.lens;           // lens 大的在前
        return a.his_lens > b.his_lens;       // lens 相同则 his_lens 大的在前
    };
    std::sort(prefill_reqs.begin(), prefill_reqs.end(), sort_by_lens_desc);
    std::sort(decode_reqs.begin(), decode_reqs.end(), sort_by_lens_desc);

    res_reqs.insert(res_reqs.end(), prefill_reqs.begin(), prefill_reqs.end());
    res_reqs.insert(res_reqs.end(), decode_reqs.begin(), decode_reqs.end());

    // Step 5: 拆分 micro-batches
    for (int i = 0; i < batch_size / micro_batch_size; ++i) {
        auto start = res_reqs.begin() + i * micro_batch_size;
        auto end = start + micro_batch_size;
        result.emplace_back(start, end);
    }

    return result;
}
