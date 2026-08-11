#include <cstdlib>
#include <ctime>
#include <iostream>

#include "gpu_inference/autotuner.h"
#include "gpu_inference/build_info.h"
#include "gpu_inference/json_lite.h"

namespace gie {

std::string TuneCache::default_path() {
  if (const char* p = std::getenv("GIE_TUNE_CACHE")) return p;
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/gpu_inference/tune_cache.json";
}

TuneCache::TuneCache(std::string path) : path_(path.empty() ? default_path() : std::move(path)) {}

std::string TuneCache::make_key(const ConvProblem& prob, const DeviceInfo& dev, Backend scope) {
  // Everything that can change which kernel is fastest or valid: hardware, software stack,
  // operator definition (geometry, precision, fusion) and the library version.
  return dev.name + "|sm" + std::to_string(dev.cc()) + "|sms" + std::to_string(dev.sm_count) + "|cudart" +
         std::to_string(dev.runtime_version) + "|drv" + std::to_string(dev.driver_version) + "|lib" +
         std::to_string(kLibraryVersion) + "|src" + build_info().source_fingerprint + "|" + prob.key() + (scope == Backend::kAuto ? "" : std::string("|only=") + to_string(scope));
}

void TuneCache::load_locked() const {
  entries_.clear();
  loaded_ = true;
  try {
    const json::Value root = json::parse_file(path_);
    const json::Value* ents = root.find("entries");
    const json::Value* schema = root.find("schema");
    if (!ents || !ents->is_object() || !schema || schema->as_int() != 1) return;  // unknown schema: ignore
    for (const auto& kv : ents->items()) {
      const json::Value& e = kv.second;
      const json::Value *k = e.find("kernel"), *ver = e.find("kernel_version"), *us = e.find("us");
      if (!k || !k->is_string() || !ver || !us) continue;  // malformed entry: ignore
      Entry en;
      en.kernel = k->as_string();
      en.kernel_version = int(ver->as_int());
      en.us = us->as_double();
      if (const json::Value* t = e.find("tuned_seconds")) en.tuned_seconds = t->as_double();
      if (const json::Value* t = e.find("timestamp")) en.timestamp = t->as_string();
      entries_[kv.first] = en;
    }
  } catch (const Error&) {
    // Missing file => empty cache. A corrupt file is also treated as empty and rewritten on store.
  }
}

void TuneCache::save_locked() const {
  json::Value root = json::Value::object();
  root["schema"] = 1;
  root["library_version"] = kLibraryVersion;
  json::Value ents = json::Value::object();
  for (const auto& kv : entries_) {
    json::Value e = json::Value::object();
    e["kernel"] = kv.second.kernel;
    e["kernel_version"] = kv.second.kernel_version;
    e["us"] = kv.second.us;
    e["tuned_seconds"] = kv.second.tuned_seconds;
    e["timestamp"] = kv.second.timestamp;
    ents[kv.first] = std::move(e);
  }
  root["entries"] = std::move(ents);
  const size_t slash = path_.find_last_of('/');
  if (slash != std::string::npos) {
    const std::string cmd = "mkdir -p '" + path_.substr(0, slash) + "'";
    if (std::system(cmd.c_str()) != 0) throw Error(StatusCode::kIo, "cannot create cache directory for " + path_);
  }
  json::write_file(path_, root);
}

bool TuneCache::lookup(const ConvProblem& prob, const DeviceInfo& dev, std::string* kernel, double* us,
                       double* tuned_seconds, Backend scope) const {
  std::lock_guard<std::mutex> lock(mu_);
  load_locked();  // always re-read: another process may have tuned meanwhile
  auto it = entries_.find(make_key(prob, dev, scope));
  if (it == entries_.end()) return false;
  const KernelDesc* k = find_kernel(it->second.kernel);
  if (!k || k->version != it->second.kernel_version) return false;  // stale: kernel removed or changed
  std::string why;
  if (!k->supports(prob, dev, &why)) return false;
  *kernel = it->second.kernel;
  if (us) *us = it->second.us;
  if (tuned_seconds) *tuned_seconds = it->second.tuned_seconds;
  return true;
}

void TuneCache::store(const TuneResult& r, const DeviceInfo& dev, Backend scope) {
  const KernelDesc* k = find_kernel(r.best_kernel);
  if (!k) return;
  std::lock_guard<std::mutex> lock(mu_);
  load_locked();
  Entry e;
  e.kernel = r.best_kernel;
  e.kernel_version = k->version;
  e.us = r.best_us;
  e.tuned_seconds = r.search_seconds;
  const std::time_t now = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  e.timestamp = buf;
  entries_[make_key(r.problem, dev, scope)] = e;
  save_locked();
}

void TuneCache::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
  loaded_ = true;
  save_locked();
}

size_t TuneCache::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  load_locked();
  return entries_.size();
}

}  // namespace gie
