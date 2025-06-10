#include<iostream>

#define DEBUG_ENABLE
#ifdef DEBUG_ENABLE
	#define DEBUG(...) \
		do { \
			std::cout << "DEBUG [" << __FILE__ << ":" << __LINE__ << "] "; \
			DebugHelper{__VA_ARGS__}; \
			std::cout << std::endl; \
		} while (0)

	struct DebugHelper {
		template <typename... Args>
		DebugHelper(Args&&... args) {
			((std::cout << std::forward<Args>(args) << " "), ...);
		}
	};
#else
    #define DEBUG(...) \
        do { \
        } while (0)
#endif