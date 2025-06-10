#include "nns/nns.h"
#include "compass/request_generator.h"
#include "debug.h"

int main() {
    ReqGenerator generator(64);
    auto batches = generator.generateReq(8);
    //auto n=create_GPT3(batches[0],1,256,8,32);
    //6.7B
    //auto n=create_GPT3(batches[0],32,4096,32,128);
    //13B
    auto n=create_GPT3(batches[0],40,5120,40,128);
    //175B
    //auto n=create_GPT3(batches[0],96,12288,96,128);
    DEBUG(n->len());
    
    return 0;
}