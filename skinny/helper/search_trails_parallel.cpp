#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
 
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
 
#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif
static_assert(NUM_THREADS >= 1, "NUM_THREADS must be >= 1");
 
namespace py = pybind11;

namespace {

constexpr int MAX_KEY_WORDS = 32;  // generous cap: covers sbox_size up to 32 with key_size up to 3

// SKINNY's 6-bit round constants.  AddConstants is affine: it does not
// change mask propagation or a trail's correlation magnitude, but it adds
// the phase (-1)^<v,c>, where v is the mask on the output of SubCells (and
// therefore on the input of AddConstants).  The accelerated path must apply
// this phase for every linear combination enumerated below.
constexpr std::array<uint8_t, 62> SKINNY_ROUND_CONSTANTS = {
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B,
    0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30,
    0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
    0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
    0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A,
    0x15, 0x2A, 0x14, 0x28, 0x10, 0x20,
};

inline int parity_sign(uint64_t value) {
    return (__builtin_popcountll(static_cast<unsigned long long>(value)) & 1) ? -1 : 1;
}

// Return the AddConstants phase for one cell in one round.  Constants are
// XORed into column 0 of rows 0, 1, and 2 exactly as in the Python reference.
inline int round_constant_phase_sign(std::size_t round_index, uint64_t v, int r, int c) {
    if (c != 0) return 1;

    const uint64_t round_constant = SKINNY_ROUND_CONSTANTS[round_index];
    if (r == 0) return parity_sign(v & (round_constant & 0x0FULL));
    if (r == 1) return parity_sign(v & (round_constant >> 4));
    if (r == 2) return parity_sign(v & 0x02ULL);
    return 1;
}

// Parse a big-endian byte string into `expected_words` 64-bit words.
std::vector<uint64_t> bytes_to_words(const std::string &bytes, std::size_t expected_words) {
    if (bytes.size() != expected_words * 8) {
        throw std::runtime_error(
            "search_trails_in_space_cpp: expected a " + std::to_string(expected_words * 8) +
            "-byte big-endian integer, got " + std::to_string(bytes.size()) + " bytes");
    }
    std::vector<uint64_t> words(expected_words);
    const unsigned char *p = reinterpret_cast<const unsigned char *>(bytes.data());
    for (std::size_t w = 0; w < expected_words; ++w) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) v = (v << 8) | p[w * 8 + b];
        words[w] = v;
    }
    return words;
}
 
// Serialize 64-bit words back to a big-endian byte string.
std::string words_to_bytes(const uint64_t *words, std::size_t n) {
    std::string out(n * 8, '\0');
    unsigned char *p = reinterpret_cast<unsigned char *>(&out[0]);
    for (std::size_t w = 0; w < n; ++w) {
        uint64_t v = words[w];
        for (int b = 7; b >= 0; --b) {
            p[w * 8 + b] = static_cast<unsigned char>(v & 0xFF);
            v >>= 8;
        }
    }
    return out;
}
 
// Combined (plaintext_mask << 64*key_size_bits) + key_mask key, as up to
// MAX_KEY_WORDS 64-bit words.
struct KeyWords {
    std::array<uint64_t, MAX_KEY_WORDS> w{};
    int n = 0;
    bool operator==(const KeyWords &o) const {
        if (n != o.n) return false;
        for (int i = 0; i < n; ++i)
            if (w[i] != o.w[i]) return false;
        return true;
    }
};
struct KeyWordsHash {
    std::size_t operator()(const KeyWords &k) const {
        std::size_t h = 1469598103934665603ULL;  // FNV-1a
        for (int i = 0; i < k.n; ++i) {
            h ^= k.w[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};
 
// Derived layout parameters for a given sbox_size, shared by the real
// entry point and the debug/unit-test helpers below.
struct Layout {
    int sbox_size;
    int cells_per_word;  // how many sbox_size-bit cells fit in one 64-bit word
    int words_per_plane;  // how many 64-bit words a whole (4,4,sbox_size) plane needs
    uint64_t mask;
    uint64_t size;  // 2**sbox_size -- the qdtm index range for a single mask/diff value
 
    static Layout make(int sbox_size) {
        if (sbox_size < 1 || sbox_size > 32)
            throw std::runtime_error("search_trails_in_space_cpp: sbox_size out of supported range (1..32)");
        if (64 % sbox_size != 0)
            throw std::runtime_error(
                "search_trails_in_space_cpp: sbox_size=" + std::to_string(sbox_size) +
                " does not evenly divide 64 -- a cell would straddle two 64-bit words, which this "
                "implementation does not support (sbox_size=4 and 8, the two real SKINNY variants, both work)");
        int cells_per_word = 64 / sbox_size;
        if (16 % cells_per_word != 0)
            throw std::runtime_error(
                "search_trails_in_space_cpp: sbox_size=" + std::to_string(sbox_size) +
                " does not give a whole number of 64-bit words per (4,4)-plane");
        Layout L;
        L.sbox_size = sbox_size;
        L.cells_per_word = cells_per_word;
        L.words_per_plane = 16 / cells_per_word;
        L.mask = (sbox_size == 64) ? ~0ULL : ((uint64_t{1} << sbox_size) - 1);
        L.size = uint64_t{1} << sbox_size;
        return L;
    }
};
 
// Extract the sbox_size-bit value at grid position (r,c) out of a plane
// stored as `layout.words_per_plane` consecutive 64-bit words (MSB-first,
// row-major over (r,c), matching helper/utils.py::convert_int_to_array's
// bit order).
inline uint64_t extract_cell(const uint64_t *plane_words, const Layout &layout, int r, int c) {
    int cell_index = r * 4 + c;  // 0..15
    int word_idx = cell_index / layout.cells_per_word;
    int pos_in_word = cell_index % layout.cells_per_word;
    int shift = 64 - (pos_in_word + 1) * layout.sbox_size;
    return (plane_words[word_idx] >> shift) & layout.mask;
}
 
}  // namespace
 
py::list search_trails_in_space_cpp(
    py::list state_bytes_list,      // solution_set[i].state_mask_int_vars.to_bytes(2*sbox_size, 'big')
    py::list key_bytes_list,        // solution_set[i].keys_mask_mod2_int_vars.to_bytes(2*sbox_size*key_size, 'big')
    py::list state_all_bytes_list,  // solution_set[i].state_all_mask_int_vars.to_bytes(2*sbox_size*3*nr, 'big')
    py::list before_sb,             // nr lists of 16 ints (0..size-1) -- diff_trail.rounds[n].beforeSB
    py::list after_sb,              // nr lists of 16 ints (0..size-1) -- diff_trail.rounds[n].afterSB
    int nr,
    int sbox_size,
    int key_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> qdtm  // (size*size, size*size), skinny.qdtm
) {
    const Layout layout = Layout::make(sbox_size);
    if (nr <= 0) throw std::runtime_error("search_trails_in_space_cpp: nr must be positive");
    if (static_cast<std::size_t>(nr) > SKINNY_ROUND_CONSTANTS.size())
        throw std::runtime_error("search_trails_in_space_cpp: nr exceeds the SKINNY round-constant table");
    if (key_size < 1 || key_size > 3)
        throw std::runtime_error("search_trails_in_space_cpp: key_size must be 1..3");
 
    const std::size_t dim = state_bytes_list.size();
    if (key_bytes_list.size() != dim || state_all_bytes_list.size() != dim)
        throw std::runtime_error("search_trails_in_space_cpp: solution_set arrays have mismatched lengths");
    if (dim > 62)
        throw std::runtime_error("search_trails_in_space_cpp: dim=" + std::to_string(dim) +
                                  " -- 2**dim subsets is not computable in any implementation");
 
    const int wpp = layout.words_per_plane;
    const std::size_t state_words_n = static_cast<std::size_t>(wpp);
    const std::size_t key_words_n = static_cast<std::size_t>(wpp) * key_size;
    const std::size_t state_all_words_n = static_cast<std::size_t>(wpp) * 3 * nr;
    const int combined_words_n = wpp * (1 + key_size);
    if (combined_words_n > MAX_KEY_WORDS)
        throw std::runtime_error("search_trails_in_space_cpp: combined key needs " +
                                  std::to_string(combined_words_n) + " 64-bit words, exceeding the compiled-in cap of " +
                                  std::to_string(MAX_KEY_WORDS));
 
    std::vector<std::vector<uint64_t>> state_word(dim);
    std::vector<std::vector<uint64_t>> key_words(dim);
    std::vector<std::vector<uint64_t>> state_all_words(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        state_word[i] = bytes_to_words(state_bytes_list[i].cast<std::string>(), state_words_n);
        key_words[i] = bytes_to_words(key_bytes_list[i].cast<std::string>(), key_words_n);
        state_all_words[i] = bytes_to_words(state_all_bytes_list[i].cast<std::string>(), state_all_words_n);
    }
 
    std::vector<std::array<int, 16>> a_before(nr), a_after(nr);
    for (int n = 0; n < nr; ++n) {
        auto brow = before_sb[n].cast<std::vector<int>>();
        auto arow = after_sb[n].cast<std::vector<int>>();
        if (brow.size() != 16 || arow.size() != 16)
            throw std::runtime_error("search_trails_in_space_cpp: beforeSB/afterSB rows must have 16 entries");
        for (int k = 0; k < 16; ++k) {
            a_before[n][k] = brow[k];
            a_after[n][k] = arow[k];
        }
    }
 
    auto Q = qdtm.unchecked<2>();
    const int64_t expected_dim = static_cast<int64_t>(layout.size) * static_cast<int64_t>(layout.size);
    if (Q.shape(0) != expected_dim || Q.shape(1) != expected_dim)
        throw std::runtime_error("search_trails_in_space_cpp: qdtm must be " + std::to_string(expected_dim) + "x" +
                                  std::to_string(expected_dim) + " for sbox_size=" + std::to_string(sbox_size));
 
    const uint64_t size = layout.size;
 
    // Evaluates one subset's correlation and, if nonzero, accumulates it
    // into `local_result`. Reused identically by every thread -- pure
    // function of its arguments, touches no shared mutable state, so it's
    // safe to call concurrently from multiple threads as long as each
    // gets its own `local_result` map (which is exactly how it's used
    // below).
    auto process_subset = [&](const std::vector<uint64_t> &state, const std::vector<uint64_t> &keym,
                               const std::vector<uint64_t> &stateall,
                               std::unordered_map<KeyWords, double, KeyWordsHash> &local_result) {
        int sign = 1;
        // QDTM entries are signed dyadic rationals. Multiply them directly:
        // log2/exp2 round trips introduce errors even when the product is
        // exactly representable, which can turn exact FWHT zeros negative.
        double value = 1.0;
        bool aborted = false;
        for (int n = 0; n < nr && !aborted; ++n) {
            const uint64_t *wu = &stateall[(static_cast<std::size_t>(n) * 3 + 0) * wpp];
            const uint64_t *wv = &stateall[(static_cast<std::size_t>(n) * 3 + 1) * wpp];

            // AddConstants follows SubCells, so its phase is determined by
            // the SubCells output mask v.  Only column 0 of rows 0..2 is
            // affected; compute those three factors once per round rather
            // than branching in every S-box iteration below.
            sign *= round_constant_phase_sign(
                static_cast<std::size_t>(n), extract_cell(wv, layout, 0, 0), 0, 0);
            sign *= round_constant_phase_sign(
                static_cast<std::size_t>(n), extract_cell(wv, layout, 1, 0), 1, 0);
            sign *= round_constant_phase_sign(
                static_cast<std::size_t>(n), extract_cell(wv, layout, 2, 0), 2, 0);

            for (int r = 0; r < 4 && !aborted; ++r) {
                for (int c = 0; c < 4; ++c) {
                    int a = a_before[n][4 * r + c];
                    int bdiff = a_after[n][4 * r + c];
                    uint64_t u = extract_cell(wu, layout, r, c);
                    uint64_t v = extract_cell(wv, layout, r, c);
                    double val = Q(static_cast<int64_t>(v * size + bdiff), static_cast<int64_t>(u * size + a));
                    if (val == 0.0) {
                        aborted = true;
                        break;
                    }
                    value *= val;
                }
            }
        }
        if (aborted) return;
 
        value *= sign;
 
        KeyWords kw;
        kw.n = combined_words_n;
        for (int t = 0; t < wpp; ++t) kw.w[t] = state[t];
        for (int t = 0; t < static_cast<int>(key_words_n); ++t) kw.w[wpp + t] = keym[t];
 
        local_result[kw] += value;
    };
 
    const uint64_t total = (dim == 0) ? 1ULL : (uint64_t{1} << dim);
    const int actual_threads = static_cast<int>(std::min<uint64_t>(NUM_THREADS, total));
 
    // Split [0, total) into `actual_threads` contiguous chunks (as even as
    // possible; any remainder is spread one-per-chunk over the first
    // chunks) and give each its own result map to accumulate into.
    std::vector<std::pair<uint64_t, uint64_t>> ranges(actual_threads);
    {
        uint64_t chunk = total / static_cast<uint64_t>(actual_threads);
        uint64_t rem = total % static_cast<uint64_t>(actual_threads);
        uint64_t cursor = 0;
        for (int t = 0; t < actual_threads; ++t) {
            uint64_t sz = chunk + (static_cast<uint64_t>(t) < rem ? 1 : 0);
            ranges[t] = {cursor, cursor + sz};
            cursor += sz;
        }
    }
    std::vector<std::unordered_map<KeyWords, double, KeyWordsHash>> thread_results(actual_threads);
 
    auto worker = [&](int t) {
        uint64_t start = ranges[t].first;
        uint64_t end = ranges[t].second;
        if (start >= end) return;  // can happen if total < actual_threads (shouldn't, given the min() above)
 
        // Seed this thread's accumulator to the state the single-threaded
        // algorithm would have reached right before processing index
        // `start`: XOR of the solutions selected by gray(start)'s set bits.
        std::vector<uint64_t> acc_state(state_words_n, 0);
        std::vector<uint64_t> acc_key(key_words_n, 0);
        std::vector<uint64_t> acc_state_all(state_all_words_n, 0);
 
        uint64_t g = start ^ (start >> 1);
        for (int b = 0; b < static_cast<int>(dim); ++b) {
            if ((g >> b) & 1ULL) {
                for (std::size_t w = 0; w < acc_state.size(); ++w) acc_state[w] ^= state_word[b][w];
                for (std::size_t w = 0; w < acc_key.size(); ++w) acc_key[w] ^= key_words[b][w];
                for (std::size_t w = 0; w < acc_state_all.size(); ++w) acc_state_all[w] ^= state_all_words[b][w];
            }
        }
 
        auto &local_result = thread_results[t];
        process_subset(acc_state, acc_key, acc_state_all, local_result);
 
        for (uint64_t i = start + 1; i < end; ++i) {
            // Gray-code single-bit toggle: bit = index of the lowest set bit of i.
            std::size_t sol = static_cast<std::size_t>(__builtin_ctzll(i));
            for (std::size_t w = 0; w < acc_state.size(); ++w) acc_state[w] ^= state_word[sol][w];
            for (std::size_t w = 0; w < acc_key.size(); ++w) acc_key[w] ^= key_words[sol][w];
            for (std::size_t w = 0; w < acc_state_all.size(); ++w) acc_state_all[w] ^= state_all_words[sol][w];
 
            process_subset(acc_state, acc_key, acc_state_all, local_result);
        }
    };
 
    {
        // The worker threads touch no Python objects at all (only plain
        // std::vector/array data extracted above), so it's safe -- and
        // necessary for real parallelism -- to release the GIL for the
        // duration of the computation.
        py::gil_scoped_release release;
#if NUM_THREADS > 1
        std::vector<std::thread> threads;
        threads.reserve(actual_threads);
        for (int t = 0; t < actual_threads; ++t) threads.emplace_back(worker, t);
        for (auto &th : threads) th.join();
#else
        worker(0);
#endif
    }
 
    std::unordered_map<KeyWords, double, KeyWordsHash> result;
    for (auto &tr : thread_results)
        for (auto &kv : tr) result[kv.first] += kv.second;
 
    py::list out;
    for (auto &kv : result)
        out.append(py::make_tuple(py::bytes(words_to_bytes(kv.first.w.data(), kv.first.n)), kv.second));
    return out;
}
 
// ---------------------------------------------------------------------------
// Debug/unit-test-only helpers: expose the word-layout math and per-cell
// extraction in isolation, so the generalized sbox_size handling can be
// tested directly against a Python bit-string reference without needing a
// (size*size)x(size*size) qdtm -- which, for sbox_size=8, would be a dense
// 32 GiB array and isn't something a correctness test should have to
// allocate. Not part of the public API used by search_trails_in_space.
// ---------------------------------------------------------------------------
 
int _debug_words_per_plane(int sbox_size) { return Layout::make(sbox_size).words_per_plane; }
 
uint64_t _debug_extract_cell(py::bytes plane_bytes, int sbox_size, int r, int c) {
    const Layout layout = Layout::make(sbox_size);
    std::string s = plane_bytes;
    auto words = bytes_to_words(s, static_cast<std::size_t>(layout.words_per_plane));
    return extract_cell(words.data(), layout, r, c);
}

int _debug_round_constant_phase_sign(int round_index, uint64_t v, int r, int c) {
    if (round_index < 0 || static_cast<std::size_t>(round_index) >= SKINNY_ROUND_CONSTANTS.size())
        throw std::runtime_error("_debug_round_constant_phase_sign: round index out of range");
    if (r < 0 || r >= 4 || c < 0 || c >= 4)
        throw std::runtime_error("_debug_round_constant_phase_sign: cell index out of range");
    return round_constant_phase_sign(static_cast<std::size_t>(round_index), v, r, c);
}

PYBIND11_MODULE(skinny_cpp, m) {
    m.doc() = "C++ acceleration for search_trails_in_space";
    m.def("search_trails_in_space_cpp", &search_trails_in_space_cpp,
          py::arg("state_bytes_list"), py::arg("key_bytes_list"), py::arg("state_all_bytes_list"),
          py::arg("before_sb"), py::arg("after_sb"), py::arg("nr"), py::arg("sbox_size"), py::arg("key_size"),
          py::arg("qdtm"));
    m.def("_debug_words_per_plane", &_debug_words_per_plane);
    m.def("_debug_extract_cell", &_debug_extract_cell);
    m.def("_debug_round_constant_phase_sign", &_debug_round_constant_phase_sign);
}
