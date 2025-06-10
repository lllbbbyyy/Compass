#include <vector>

#include "tqdm/tqdm.h"


int main() {
	std::vector<int> v(100000000,0);
	for (int i : tqdm(v.size())) {
		v[i]++;
	}
	for (int i : tqdm(v)) {
		v[i]++;
	}
	return 0;
}