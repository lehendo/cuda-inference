#include "gpu_inference/json_lite.h"

#include <unistd.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "gpu_inference/error.h"

namespace gie {
namespace json {

Value& Value::operator[](const std::string& key) {
  if (type_ == Type::kNull) type_ = Type::kObject;
  for (auto& kv : obj_)
    if (kv.first == key) return kv.second;
  obj_.emplace_back(key, Value());
  return obj_.back().second;
}

const Value* Value::find(const std::string& key) const {
  for (const auto& kv : obj_)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

namespace {

void escape(const std::string& s, std::string& out) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  out += '"';
}

}  // namespace

void Value::dump_to(std::string& out, int indent, int level) const {
  const bool pretty = indent >= 0;
  auto nl = [&](int lvl) {
    if (pretty) {
      out += '\n';
      out.append(size_t(lvl * indent), ' ');
    }
  };
  switch (type_) {
    case Type::kNull: out += "null"; break;
    case Type::kBool: out += bool_ ? "true" : "false"; break;
    case Type::kNumber: {
      if (!std::isfinite(num_)) { out += "null"; break; }
      char buf[40];
      if (num_ == std::floor(num_) && std::fabs(num_) < 9e15) std::snprintf(buf, sizeof(buf), "%lld", (long long)num_);
      else std::snprintf(buf, sizeof(buf), "%.17g", num_);
      out += buf;
      break;
    }
    case Type::kString: escape(str_, out); break;
    case Type::kArray:
      if (arr_.empty()) { out += "[]"; break; }
      out += '[';
      for (size_t i = 0; i < arr_.size(); ++i) {
        if (i) out += ',';
        nl(level + 1);
        arr_[i].dump_to(out, indent, level + 1);
      }
      nl(level);
      out += ']';
      break;
    case Type::kObject:
      if (obj_.empty()) { out += "{}"; break; }
      out += '{';
      for (size_t i = 0; i < obj_.size(); ++i) {
        if (i) out += ',';
        nl(level + 1);
        escape(obj_[i].first, out);
        out += pretty ? ": " : ":";
        obj_[i].second.dump_to(out, indent, level + 1);
      }
      nl(level);
      out += '}';
      break;
  }
}

std::string Value::dump(int indent) const {
  std::string s;
  dump_to(s, indent, 0);
  return s;
}

namespace {

class Parser {
 public:
  explicit Parser(const std::string& t) : s_(t) {}
  Value parse_document() {
    Value v = parse_value();
    skip_ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& m) {
    int line = 1, col = 1;
    for (size_t k = 0; k < i_ && k < s_.size(); ++k) {
      if (s_[k] == '\n') { ++line; col = 1; } else { ++col; }
    }
    throw Error(StatusCode::kIo, "JSON parse error at line " + std::to_string(line) + ", column " +
                                     std::to_string(col) + ": " + m);
  }
  void skip_ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  char peek() {
    skip_ws();
    if (i_ >= s_.size()) fail("unexpected end of input");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++i_;
  }
  Value parse_value() {
    const char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Value(parse_string());
    if (c == 't') return literal("true", Value(true));
    if (c == 'f') return literal("false", Value(false));
    if (c == 'n') return literal("null", Value());
    return parse_number();
  }
  Value literal(const char* word, Value v) {
    const size_t n = std::string(word).size();
    if (s_.compare(i_, n, word) != 0) fail("invalid literal");
    i_ += n;
    return v;
  }
  Value parse_number() {
    const size_t start = i_;
    if (s_[i_] == '-') ++i_;
    while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' || s_[i_] == 'e' ||
                              s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-'))
      ++i_;
    if (start == i_) fail("invalid value");
    char* end = nullptr;
    const std::string tok = s_.substr(start, i_ - start);
    const double d = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0') fail("invalid number '" + tok + "'");
    return Value(d);
  }
  std::string parse_string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("bad escape");
        const char e = s_[i_++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) fail("bad \\u escape");
            const unsigned cp = unsigned(std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16));
            i_ += 4;
            if (cp < 0x80) out += char(cp);
            else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
            else { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
            break;
          }
          default: fail("bad escape");
        }
      } else {
        out += c;
      }
    }
    if (i_ >= s_.size()) fail("unterminated string");
    ++i_;
    return out;
  }
  Value parse_array() {
    expect('[');
    Value a = Value::array();
    if (peek() == ']') { ++i_; return a; }
    while (true) {
      a.push_back(parse_value());
      if (peek() == ',') { ++i_; continue; }
      expect(']');
      return a;
    }
  }
  Value parse_object() {
    expect('{');
    Value o = Value::object();
    if (peek() == '}') { ++i_; return o; }
    while (true) {
      skip_ws();
      const std::string key = parse_string();
      expect(':');
      o[key] = parse_value();
      if (peek() == ',') { ++i_; continue; }
      expect('}');
      return o;
    }
  }
  const std::string& s_;
  size_t i_ = 0;
};

}  // namespace

Value parse(const std::string& text) { return Parser(text).parse_document(); }

Value parse_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(StatusCode::kIo, "cannot open '" + path + "'");
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

void write_file(const std::string& path, const Value& v, int indent) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) throw Error(StatusCode::kIo, "cannot write '" + tmp + "'");
    f << v.dump(indent) << "\n";
    if (!f) throw Error(StatusCode::kIo, "write failed for '" + tmp + "'");
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw Error(StatusCode::kIo, "cannot rename onto '" + path + "'");
}

}  // namespace json
}  // namespace gie
