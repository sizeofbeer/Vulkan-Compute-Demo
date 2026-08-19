// common/i18n.h
// 极简双语支持：默认输出中文；通过命令行 --lang en / -l en 或环境变量
// COURSE_LANG=en 切换为英文。
//
// 用法：把每一个「静态中文串」用 i18n::tr("中文", "English") 包裹即可，
// 变量照常拼接在 << 链里。例如：
//   std::cout << i18n::tr("[ch01] 完成 N=", "[ch01] done N=") << N
//             << i18n::tr(" 错误数=", " errors=") << errs << "\n";
// 翻译表就近写在调用处，无需集中维护，也不会出现查不到的兜底问题。
#pragma once
#include <string>

namespace i18n {
    extern std::string lang;  // "zh" 或 "en"

    // 必须在 main() 开头调用，解析 --lang / -l 与 COURSE_LANG 环境变量。
    void init(int argc, char** argv);

    // 双参 inline：lang==en 时返回英文，否则返回中文。
    inline const std::string& tr(const std::string& zh, const std::string& en) {
        return (lang == "en") ? en : zh;
    }
}
