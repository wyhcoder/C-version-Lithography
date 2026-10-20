#pragma once

#include <filesystem>
#include <fstream>
#include <streambuf>
#include <string>

namespace litho {

// 输入完整日志路径：将 cout / cerr 同时写入原终端和文件；对象退出时刷新并恢复输出通道。
// 每次创建覆盖同名日志；调用方先创建结果目录，并让对象存活到异常信息输出之后。
class ConsoleLog final {
public:
    explicit ConsoleLog(const std::filesystem::path& path);
    ~ConsoleLog();

private:
    // 分别保留 cout、cerr 的终端目标，两者共享同一个日志文件。
    class TeeBuffer final : public std::streambuf {
    public:
        TeeBuffer(std::streambuf* terminal, std::streambuf* log);

    protected:
        int_type overflow(int_type ch) override;
        std::streamsize xsputn(const char* text, std::streamsize count) override;
        int sync() override;

    private:
        std::streambuf* terminal_; // 原终端缓冲区。
        std::streambuf* log_;      // console_output.log 的文件缓冲区。
    };

    std::ofstream file_; // 最后销毁文件，确保恢复终端之前日志目标仍然有效。
    std::streambuf* original_out_;
    std::streambuf* original_err_;
    TeeBuffer out_;
    TeeBuffer err_;
};

// 输入已经完成路径引用/转义的绘图命令；合并子进程的标准/错误输出，转写到 cout 的终端与日志。
// 返回子进程结束状态，0 表示成功；用于保存 LSM Python 绘图脚本的实际输出。
int run_logged_command(const std::string& command);

} // namespace litho
