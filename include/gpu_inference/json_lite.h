// Small self-contained JSON value/parser/serializer (no third-party dependency).
// Supports the full JSON grammar needed for the tuning cache, workload files and
// result files: null, bool, number (double / int64), string, array, object.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gie {
namespace json {

class Value {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : type_(Type::kBool), bool_(b) {}
  Value(int v) : type_(Type::kNumber), num_(v) {}
  Value(int64_t v) : type_(Type::kNumber), num_(double(v)) {}
  Value(double v) : type_(Type::kNumber), num_(v) {}
  Value(const char* s) : type_(Type::kString), str_(s) {}
  Value(std::string s) : type_(Type::kString), str_(std::move(s)) {}

  static Value array() { Value v; v.type_ = Type::kArray; return v; }
  static Value object() { Value v; v.type_ = Type::kObject; return v; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_object() const { return type_ == Type::kObject; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_number() const { return type_ == Type::kNumber; }

  bool as_bool(bool def = false) const { return type_ == Type::kBool ? bool_ : def; }
  double as_double(double def = 0) const { return type_ == Type::kNumber ? num_ : def; }
  int64_t as_int(int64_t def = 0) const { return type_ == Type::kNumber ? int64_t(num_) : def; }
  const std::string& as_string() const { return str_; }

  // Object access (insertion order preserved). operator[] inserts; find() returns null when absent.
  Value& operator[](const std::string& key);
  const Value* find(const std::string& key) const;
  bool contains(const std::string& key) const { return find(key) != nullptr; }
  const std::vector<std::pair<std::string, Value>>& items() const { return obj_; }

  // Array access.
  void push_back(Value v) { arr_.push_back(std::move(v)); }
  const std::vector<Value>& elements() const { return arr_; }
  size_t size() const { return type_ == Type::kArray ? arr_.size() : obj_.size(); }

  std::string dump(int indent = -1) const;  // indent < 0: compact

 private:
  void dump_to(std::string& out, int indent, int level) const;
  Type type_ = Type::kNull;
  bool bool_ = false;
  double num_ = 0;
  std::string str_;
  std::vector<Value> arr_;
  std::vector<std::pair<std::string, Value>> obj_;
};

// Throws gie::Error(kIo) with line/column on malformed input.
Value parse(const std::string& text);
Value parse_file(const std::string& path);
void write_file(const std::string& path, const Value& v, int indent = 2);  // atomic (tmp + rename)

}  // namespace json
}  // namespace gie
