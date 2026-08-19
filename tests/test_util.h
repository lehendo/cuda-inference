// Tiny dependency-free test harness: TEST(name) { ... }, CHECK(cond), CHECK_EQ, run via main().
#pragma once

#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gt {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};
inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}
inline int& failures() {
  static int f = 0;
  return f;
}
inline int& checks() {
  static int c = 0;
  return c;
}
struct Registrar {
  Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::cerr << "  FAIL " << file << ":" << line << ": " << msg << "\n";
}

}  // namespace gt

#define GT_CONCAT2(a, b) a##b
#define GT_CONCAT(a, b) GT_CONCAT2(a, b)
#define TEST(name)                                                              \
  static void GT_CONCAT(test_fn_, name)();                                      \
  static gt::Registrar GT_CONCAT(test_reg_, name)(#name, GT_CONCAT(test_fn_, name)); \
  static void GT_CONCAT(test_fn_, name)()

#define CHECK(cond)                                                \
  do {                                                             \
    ++gt::checks();                                                \
    if (!(cond)) gt::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_MSG(cond, msg)                                                   \
  do {                                                                         \
    ++gt::checks();                                                            \
    if (!(cond)) {                                                             \
      std::ostringstream gt_os;                                                \
      gt_os << "CHECK(" #cond ") " << msg;                                     \
      gt::fail(__FILE__, __LINE__, gt_os.str());                               \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b)                                                                    \
  do {                                                                                    \
    ++gt::checks();                                                                       \
    if (!((a) == (b))) {                                                                  \
      std::ostringstream gt_os;                                                           \
      gt_os << "CHECK_EQ(" #a ", " #b ") got " << (a) << " vs " << (b);                   \
      gt::fail(__FILE__, __LINE__, gt_os.str());                                          \
    }                                                                                     \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                             \
  do {                                                                                    \
    ++gt::checks();                                                                       \
    if (!(std::fabs(double(a) - double(b)) <= double(tol))) {                             \
      std::ostringstream gt_os;                                                           \
      gt_os << "CHECK_NEAR(" #a ", " #b ") got " << (a) << " vs " << (b);                 \
      gt::fail(__FILE__, __LINE__, gt_os.str());                                          \
    }                                                                                     \
  } while (0)

// Expects `expr` to throw gie::Error with the given status code.
#define CHECK_THROWS_CODE(expr, status_code)                                              \
  do {                                                                                    \
    ++gt::checks();                                                                       \
    bool gt_threw = false;                                                                \
    try {                                                                                 \
      expr;                                                                               \
    } catch (const gie::Error& gt_e) {                                                    \
      gt_threw = gt_e.code() == (status_code);                                            \
      if (!gt_threw) std::cerr << "    (threw " << gt_e.what() << ")\n";                  \
    } catch (...) {                                                                       \
    }                                                                                     \
    if (!gt_threw) gt::fail(__FILE__, __LINE__, "expected gie::Error code for: " #expr);  \
  } while (0)

inline int gt_main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (auto& t : gt::registry()) {
    if (filter && t.name.find(filter) == std::string::npos) continue;
    const int before = gt::failures();
    std::cout << "[ RUN  ] " << t.name << std::endl;
    try {
      t.fn();
    } catch (const std::exception& e) {
      gt::fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    }
    std::cout << (gt::failures() == before ? "[  OK  ] " : "[ FAIL ] ") << t.name << std::endl;
    ++ran;
  }
  std::cout << ran << " tests, " << gt::checks() << " checks, " << gt::failures() << " failures\n";
  if (ran == 0) {
    std::cerr << "no tests matched\n";
    return 1;
  }
  return gt::failures() == 0 ? 0 : 1;
}

#define GT_MAIN() \
  int main(int argc, char** argv) { return gt_main(argc, argv); }
