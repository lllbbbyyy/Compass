// req_generator.h
#ifndef COMPASS_REQUEST_GENERATOR_H
#define COMPASS_REQUEST_GENERATOR_H

#include <vector>
#include <utility>
#include <optional>
#include <string>
#include "json.hpp"
#include <fstream>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>

class Req {
public:
    // 定义请求类型枚举
    enum class Type {
        Prefill,
        Decode
    };
    int id;
    Type type; 
    int lens;
    int his_lens;

    Req(int req_id, Type req_type, int req_lens, int req_his_lens);
    
    friend std::ostream& operator<<(std::ostream& os, const Req& req) {
        std::string type_str;
        switch (req.type) {
            case Type::Prefill:
                type_str = "Prefill";
                break;
            case Type::Decode:
                type_str = "Decode";
                break;
        }
        os << "id:" << req.id 
           << ",type:" << type_str 
           << ",lens:" << req.lens 
           << ",his_lens:" << req.his_lens;
        return os;
    }
};

using batchedReqs_t=std::vector<std::vector<Req>>;

class ReqGenerator {
public:
    // const char* inputLengthsFile="../config/simulated_input_lengths.json";
    // const char* outputLengthsFile="../config/simulated_output_lengths.json";
    static std::string inputLengthsFile;
    static std::string outputLengthsFile;

    ReqGenerator(int batch_size);
    batchedReqs_t generateReq(int micro_batch_size);
    batchedReqs_t generateReq(int micro_batch_size,int num_prefill, int num_decode);

private:
    int now_id;
    std::vector<std::pair<int, std::optional<Req>>> req_cache;
    int batch_size;
    std::vector<Req> res_reqs;

    std::vector<int> simulated_input_lengths;
    std::vector<int> simulated_output_lengths;
    size_t input_index;
    size_t output_index;

    void warmupGenerate();
    int getNextInputLength();
    int getNextOutputLength();
};

#endif // COMPASS_REQUEST_GENERATOR_H