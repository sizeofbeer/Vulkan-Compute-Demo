// common/i18n.cpp
#include "i18n.h"
#include <cstdlib>
#include <string>

namespace i18n {
    std::string lang = "zh";

    void init(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if ((a == "--lang" || a == "-l") && i + 1 < argc) {
                lang = argv[++i];
            }
        }
        // 命令行优先；命令行没给时再读环境变量，方便 CI / 脚本统一切换。
        if (lang == "zh") {
            if (const char* e = std::getenv("COURSE_LANG")) {
                if (std::string(e) == "en") lang = "en";
            }
        }
    }
}
