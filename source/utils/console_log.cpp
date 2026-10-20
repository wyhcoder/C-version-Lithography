#include "console_log.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>

namespace litho {

ConsoleLog::TeeBuffer::TeeBuffer(std::streambuf* terminal, std::streambuf* log) : terminal_(terminal), log_(log) {}

ConsoleLog::TeeBuffer::int_type ConsoleLog::TeeBuffer::overflow(int_type ch) {
    if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
    const char c = traits_type::to_char_type(ch);
    const auto terminal_result = terminal_->sputc(c), log_result = log_->sputc(c);
    return traits_type::eq_int_type(terminal_result, traits_type::eof()) ||
           traits_type::eq_int_type(log_result, traits_type::eof()) ? traits_type::eof() : ch;
}

std::streamsize ConsoleLog::TeeBuffer::xsputn(const char* text, std::streamsize count) {
    const auto terminal_count = terminal_->sputn(text, count), log_count = log_->sputn(text, count);
    return std::min(terminal_count, log_count);
}

int ConsoleLog::TeeBuffer::sync() {
    const int terminal_result = terminal_->pubsync(), log_result = log_->pubsync();
    return terminal_result == 0 && log_result == 0 ? 0 : -1;
}

ConsoleLog::ConsoleLog(const std::filesystem::path& path)
    : file_(path), original_out_(std::cout.rdbuf()), original_err_(std::cerr.rdbuf()),
      out_(original_out_, file_.rdbuf()), err_(original_err_, file_.rdbuf()) {
    std::cout.rdbuf(&out_);
    std::cerr.rdbuf(&err_);
}

ConsoleLog::~ConsoleLog() {
    // 先刷新并恢复终端，再销毁文件与双路缓冲区，避免输出流引用已经释放的对象。
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(original_out_);
    std::cerr.rdbuf(original_err_);
}

int run_logged_command(const std::string& command) {
    // 子进程直接继承文件描述符时不会经过 C++ rdbuf，因此用管道把其文本转回当前 cout。
    // 缓冲区按字节分段读取；长行或 UTF-8 中文可分段，但原始字节和换行保持不变。
    std::cout.flush();
    std::FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    std::array<char, 1024> text;
    while (std::fgets(text.data(), static_cast<int>(text.size()), pipe)) std::cout << text.data() << std::flush;
    return pclose(pipe);
}

} // namespace litho
