#include <iostream>
#include <tuple>
#include <assert.h>



std::tuple<int, int, int> tile_search(int m, int k, int n, int s) {
    assert(s >= 3);
    if(m*n+n*k+m*k<=s){
        return {m,k,n};
    }
    if(m+n+m*n<=s){
        return {m,(s-m*n)/(m*n),n};
    }
    if(1+2*n<=s){
        return {(s-n)/(1+n),1,n};
    }
    return {1,1,(s-1)/2};
}

int main() {
    int m = 512, k = 512, n = 512;
    int s = 64 * 1024;  // 64KB

    auto [mtile, ktile, ntile] = tile_search(m, k, n, s);
    std::cout << "Tile found:\n";
    std::cout << "  mtile = " << mtile << "\n";
    std::cout << "  ktile = " << ktile << "\n";
    std::cout << "  ntile = " << ntile << "\n";
    std::cout<< "  buffer size = " << mtile * ktile + ktile * ntile + mtile * ntile << "\n";
    return 0;
}
