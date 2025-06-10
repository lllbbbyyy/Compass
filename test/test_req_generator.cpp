#include "compass/request_generator.h"

int main() {
    ReqGenerator generator(64);
    auto batches = generator.generateReq(64);
    
    for (const auto& batch : batches) {
        std::cout << "Batch:" << std::endl;
        for (const auto& req : batch) {
            std::cout << "  " << req << std::endl;
        }
    }
    
    return 0;
}