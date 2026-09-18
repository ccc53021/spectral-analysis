/**
 * ASCON v3 step2:
 *   load and span quasidifferential bases for every characteristic,
 *   aggregate coefficients by input-x mask,
 *   generate an x-mask basis,
 *   rank basis vectors by contribution,
 *   and save x_basis/contrib/ranked/coord_list for step3.
 *
 * Prepare:
 *   python prep_step2_data_for_c++.py
 *
 * Build:
 *   g++ -O3 -std=c++17 step2.cpp -o step2.exe
 *
 * Run:
 *   ./step2.exe output/record_trails_and_coefficients/step2_input_data.bin
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <numeric>
#include <sstream>

using namespace std;
using boost::multiprecision::cpp_int;

const int SBOX[32] = {
    0x4,0xb,0x1f,0x14,0x1a,0x15,0x9,0x2,0x1b,0x5,0x8,0x12,0x1d,0x3,0x6,0x1c,
    0x1e,0x13,0x7,0xe,0x0,0xd,0x11,0x18,0x10,0xc,0x1,0x19,0x16,0xa,0xf,0x17
};
const int ROUND_CONSTANTS[15] = {
    0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b,0x3c,0x2b,0x1a
};
const int STATE_ROWS = 5;
const int STATE_WORDS = 64;
const int STATE_BITS = STATE_ROWS * STATE_WORDS;
const uint32_t MANIFEST_MAGIC = 0x32564341;
const int MANIFEST_VERSION = 2;

using Word5 = array<uint64_t, STATE_ROWS>;
using Trail = vector<Word5>;

struct Config {
    int total_rounds = 0;
    int begin_round = 0;
    int characteristic_model_number = 0;
    int trail_model_number = 0;
    int basis_number = 0;
    int weight_range = 0;
    int dim_truncated = 0;
    string d_str;
    string mode;
    uint64_t iv = 0;

    int characteristic_len() const {
        return characteristic_model_number * total_rounds + 1;
    }
    int trail_len() const {
        return trail_model_number * total_rounds + 1;
    }
};

struct DiffData {
    double cor_rb = 0.0;
    vector<Word5> characteristic;
    string step1_path;
};

struct TrailRecord {
    int trail_id = -1;
    int weight = 0;
    int sign = 0;
    Trail trail;
};

Word5 zero_word5() {
    Word5 value{};
    value.fill(0);
    return value;
}

void clear_trail(Trail& trail) {
    for (auto& state : trail) {
        state.fill(0);
    }
}

struct Word5Hash {
    size_t operator()(const Word5& value) const noexcept {
        size_t result = 0x9e3779b97f4a7c15ULL;
        for (uint64_t word : value) {
            result ^= hash<uint64_t>{}(word)
                + 0x9e3779b97f4a7c15ULL
                + (result << 6)
                + (result >> 2);
        }
        return result;
    }
};

int read_i32(ifstream& input) {
    int32_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw runtime_error("unexpected end of manifest");
    return static_cast<int>(value);
}

string read_string(ifstream& input) {
    int length = read_i32(input);
    if (length < 0) throw runtime_error("negative string length in manifest");
    string value(static_cast<size_t>(length), '\0');
    input.read(value.data(), length);
    if (!input) throw runtime_error("unexpected end of manifest string");
    return value;
}

uint64_t read_u64(ifstream& input) {
    uint64_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw runtime_error("unexpected end of manifest uint64");
    return value;
}

Word5 read_word5(ifstream& input) {
    Word5 value{};
    for (int row = 0; row < STATE_ROWS; ++row) {
        input.read(reinterpret_cast<char*>(&value[row]), sizeof(uint64_t));
        if (!input) throw runtime_error("unexpected end of manifest Word5");
    }
    return value;
}

pair<Config, vector<DiffData>> load_manifest(const string& path) {
    ifstream input(path, ios::binary);
    if (!input) throw runtime_error("cannot open manifest: " + path);

    uint32_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!input || magic != MANIFEST_MAGIC) {
        throw runtime_error("invalid ASCON v3 manifest magic");
    }

    int version = read_i32(input);
    if (version != MANIFEST_VERSION) {
        throw runtime_error("unsupported manifest version: " + to_string(version));
    }

    Config config;
    config.total_rounds = read_i32(input);
    config.begin_round = read_i32(input);
    config.characteristic_model_number = read_i32(input);
    config.trail_model_number = read_i32(input);
    config.basis_number = read_i32(input);
    config.weight_range = read_i32(input);
    config.dim_truncated = read_i32(input);
    config.d_str = read_string(input);
    config.mode = read_string(input);
    config.iv = read_u64(input);

    if (config.total_rounds <= 0
        || config.characteristic_model_number <= 0
        || config.trail_model_number <= 0) {
        throw runtime_error("invalid model dimensions in manifest");
    }
    if (config.begin_round < 0
        || config.begin_round + config.total_rounds > 15) {
        throw runtime_error("round-constant range is invalid");
    }
    if (config.mode != "permutation"
        && config.mode != "equal"
        && config.mode != "unrestricted") {
        throw runtime_error("invalid mode in manifest: " + config.mode);
    }

    int diff_count = read_i32(input);
    if (diff_count <= 0) throw runtime_error("manifest contains no characteristics");

    vector<DiffData> diffs(static_cast<size_t>(diff_count));
    for (auto& diff : diffs) {
        input.read(reinterpret_cast<char*>(&diff.cor_rb), sizeof(diff.cor_rb));
        if (!input) throw runtime_error("unexpected end of manifest cor_rb");

        int characteristic_len = read_i32(input);
        if (characteristic_len != config.characteristic_len()) {
            throw runtime_error(
                "characteristic length mismatch: got "
                + to_string(characteristic_len)
                + ", expected "
                + to_string(config.characteristic_len())
            );
        }

        diff.characteristic.resize(static_cast<size_t>(characteristic_len));
        for (auto& state : diff.characteristic) {
            state = read_word5(input);
        }
        diff.step1_path = read_string(input);
    }

    return {config, diffs};
}

void hex_to_word5(const char* text, Word5& output) {
    if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) text += 2;

    int length = 0;
    while (length < 80 && text[length] && text[length] != '"') ++length;
    output.fill(0);

    for (int word = 0; word < STATE_ROWS; ++word) {
        int begin = length - 16 * (word + 1);
        if (begin < 0) begin = 0;
        int end = length - 16 * word;
        for (int i = begin; i < end && i < begin + 16; ++i) {
            char c = text[i];
            int digit = 0;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else throw runtime_error("invalid hex digit in trail");
            output[word] = (output[word] << 4) | static_cast<uint64_t>(digit);
        }
    }
}

bool parse_trail(const string& line, int expected_len, Trail& trail) {
    size_t position = line.find("\"trail\": [");
    if (position == string::npos) return false;
    position += 10;

    trail.assign(static_cast<size_t>(expected_len), zero_word5());
    int entry = 0;
    while (position < line.size() && entry < expected_len) {
        if (line[position] == '0'
            && position + 1 < line.size()
            && (line[position + 1] == 'x' || line[position + 1] == 'X')) {
            hex_to_word5(line.c_str() + position, trail[entry]);
            ++entry;
            while (position < line.size()
                   && line[position] != ','
                   && line[position] != ']') {
                ++position;
            }
            if (position < line.size() && line[position] == ',') ++position;
        } else {
            ++position;
        }
    }
    return entry == expected_len;
}

int parse_integer_field(const string& line, const string& field, int fallback) {
    size_t position = line.find("\"" + field + "\":");
    if (position == string::npos) return fallback;
    position += field.size() + 3;
    return stoi(line.substr(position));
}

vector<TrailRecord> load_basis_trails(
    const string& path,
    int dim,
    int expected_trail_len
) {
    ifstream input(path);
    if (!input) throw runtime_error("cannot open step1 file: " + path);

    vector<TrailRecord> records;
    string line;
    while (getline(input, line)) {
        if (line.find("\"trail_id\":") == string::npos) continue;

        int trail_id = parse_integer_field(line, "trail_id", -1);
        if (trail_id < 0) throw runtime_error("invalid trail_id in " + path);
        if (trail_id >= dim) break;

        Trail trail;
        if (!parse_trail(line, expected_trail_len, trail)) {
            throw runtime_error(
                "failed to parse a "
                + to_string(expected_trail_len)
                + "-state trail in "
                + path
            );
        }

        TrailRecord record;
        record.trail_id = trail_id;
        record.weight = parse_integer_field(line, "weight", 0);
        record.sign = parse_integer_field(line, "sign", 0);
        record.trail = move(trail);
        records.push_back(move(record));
    }

    sort(
        records.begin(),
        records.end(),
        [](const TrailRecord& left, const TrailRecord& right) {
            return left.trail_id < right.trail_id;
        }
    );
    return records;
}

int vector_inner_product(int left, int right) {
    return __builtin_parity(
        static_cast<unsigned>(left & right)
    );
}

double sbox_correlation(
    int a,
    int b,
    int u,
    int v,
    int c,
    const Config& config,
    int round,
    int column
) {
    double correlation = 0.0;
    int allowed_count = 0;
    bool conditional_first_round =
        round == 0 && config.mode != "permutation";
    int iv_bit = static_cast<int>(
        (config.iv >> (63 - column)) & 1ULL
    );

    for (int x = 0; x < 32; ++x) {
        if (conditional_first_round) {
            if (((x >> 4) & 1) != iv_bit) continue;
            if (config.mode == "equal"
                && ((x >> 1) & 1) != (x & 1)) {
                continue;
            }
        }
        ++allowed_count;

        int x1 = x ^ a;
        int y = SBOX[x ^ c];
        int y1 = SBOX[x1 ^ c];
        if ((y ^ y1) != b) continue;

        int phase = vector_inner_product(u, x)
            ^ vector_inner_product(v, y);
        correlation += phase == 0 ? 1.0 : -1.0;
    }
    if (allowed_count == 0) {
        throw runtime_error("empty allowed S-box input set");
    }
    return correlation / static_cast<double>(allowed_count);
}

double quasidifferential_correlation(
    const Config& config,
    const vector<Word5>& characteristic,
    const Trail& trail
) {
    if (static_cast<int>(characteristic.size()) != config.characteristic_len()) {
        throw runtime_error("characteristic size changed after manifest loading");
    }
    if (static_cast<int>(trail.size()) != config.trail_len()) {
        throw runtime_error("trail size does not match trail_model_number");
    }

    double correlation = 1.0;
    for (int round = 0; round < config.total_rounds; ++round) {
        const Word5& diff_in =
            characteristic[config.characteristic_model_number * round];
        const Word5& diff_out =
            characteristic[config.characteristic_model_number * round + 1];
        const Word5& mask_in =
            trail[config.trail_model_number * round];
        const Word5& mask_out =
            trail[config.trail_model_number * round + 1];
        int round_constant = ROUND_CONSTANTS[config.begin_round + round];

        for (int column = 0; column < STATE_WORDS; ++column) {
            int da = 0;
            int db = 0;
            int ma = 0;
            int mb = 0;
            for (int row = 0; row < STATE_ROWS; ++row) {
                da |= static_cast<int>((diff_in[row] >> column) & 1ULL)
                    << (4 - row);
                db |= static_cast<int>((diff_out[row] >> column) & 1ULL)
                    << (4 - row);
                ma |= static_cast<int>((mask_in[row] >> column) & 1ULL)
                    << (4 - row);
                mb |= static_cast<int>((mask_out[row] >> column) & 1ULL)
                    << (4 - row);
            }

            int c = 0;
            // Model bit i is verifier bit 63-i.  The verifier injects
            // RC << 56, hence the model uses columns 0..7, MSB-first.
            if (column < 8) {
                c = ((round_constant >> (7 - column)) & 1) << 2;
            }

            double local = sbox_correlation(
                da,
                db,
                ma,
                mb,
                c,
                config,
                round,
                column
            );
            if (local == 0.0) return 0.0;
            correlation *= local;
        }
    }
    return correlation;
}

void print_correlation_distribution(
    const map<double, uint64_t>& counts,
    const string& title
) {
    cout << '\n' << title << '\n';
    cout << "  distinct correlations: " << counts.size() << '\n';

    vector<pair<double, uint64_t>> nonzero;
    uint64_t zero_count = 0;
    for (const auto& item : counts) {
        if (item.first == 0.0) zero_count += item.second;
        else nonzero.push_back(item);
    }

    sort(
        nonzero.begin(),
        nonzero.end(),
        [](const auto& left, const auto& right) {
            long long left_weight =
                llround(-log2(abs(left.first)));
            long long right_weight =
                llround(-log2(abs(right.first)));
            if (left_weight != right_weight) return left_weight < right_weight;
            return left.first < right.first;
        }
    );

    for (const auto& item : nonzero) {
        long long weight = llround(-log2(abs(item.first)));
        cout << "  cor=" << showpos << scientific << setprecision(17)
             << item.first << noshowpos
             << "  count=" << item.second
             << "  -log2(abs(cor))=" << weight << '\n';
    }
    if (zero_count != 0) {
        cout << "  cor=" << showpos << scientific << setprecision(17)
             << 0.0 << noshowpos
             << "  count=" << zero_count
             << "  -log2(abs(cor))=infinity\n";
    }
    cout << defaultfloat;
}

void write_output(
    const Config& config,
    const vector<Word5>& x_trails,
    const vector<double>& x_coefficients
) {
    filesystem::create_directories("output/record_trails_and_coefficients");
    string path =
        "output/record_trails_and_coefficients/"
        "step2_save_trails_and_coefficients_r_"
        + to_string(config.total_rounds)
        + "_" + config.d_str
        + "_mode_" + config.mode
        + "_basis_number_" + to_string(config.basis_number)
        + "_weight_" + to_string(config.weight_range)
        + "_dim_" + to_string(config.dim_truncated)
        + ".jsonl";

    ofstream output(path);
    if (!output) throw runtime_error("cannot create output file: " + path);

    output << "{\"basis_number\":" << config.basis_number
           << ",\"weight\":" << config.weight_range
           << ",\"dim\":" << config.dim_truncated
           << ",\"x_trails\":[";

    for (size_t index = 0; index < x_trails.size(); ++index) {
        if (index != 0) output << ',';
        output << '[';
        int bit_index = 0;
        for (int row = 0; row < STATE_ROWS; ++row) {
            for (int bit = 0; bit < 64; ++bit) {
                if (bit_index++ != 0) output << ',';
                output << ((x_trails[index][row] >> bit) & 1ULL);
            }
        }
        output << ']';
    }

    output << "],\"x_coefficients\":["
           << scientific << setprecision(17);
    for (size_t index = 0; index < x_coefficients.size(); ++index) {
        if (index != 0) output << ',';
        output << x_coefficients[index];
    }
    output << "]}\n";
    cout << "Saved: " << path << '\n';
}

struct XBasis {
    vector<Word5> rows;
    vector<int> pivots;
};

bool word5_bit(const Word5& value, int bit_index) {
    return ((value[bit_index / 64] >> (bit_index % 64)) & 1ULL) != 0;
}

void xor_word5(Word5& destination, const Word5& source) {
    for (int row = 0; row < STATE_ROWS; ++row) {
        destination[row] ^= source[row];
    }
}

XBasis generate_x_basis(const vector<Word5>& masks) {
    array<Word5, STATE_BITS> pivot_rows{};
    array<bool, STATE_BITS> has_pivot{};

    for (const Word5& mask : masks) {
        Word5 candidate = mask;
        for (int column = 0; column < STATE_BITS; ++column) {
            if (!word5_bit(candidate, column)) continue;
            if (has_pivot[column]) {
                xor_word5(candidate, pivot_rows[column]);
            } else {
                pivot_rows[column] = candidate;
                has_pivot[column] = true;
                break;
            }
        }
    }

    // Convert the echelon basis to reduced row-echelon form.  Afterwards,
    // the bit at pivot[i] is exactly coordinate i, matching Sage's
    // row_space.coordinate_vector() behavior used by step2_full.py.
    for (int pivot = STATE_BITS - 1; pivot >= 0; --pivot) {
        if (!has_pivot[pivot]) continue;
        for (int earlier = 0; earlier < pivot; ++earlier) {
            if (has_pivot[earlier]
                && word5_bit(pivot_rows[earlier], pivot)) {
                xor_word5(pivot_rows[earlier], pivot_rows[pivot]);
            }
        }
    }

    XBasis result;
    for (int pivot = 0; pivot < STATE_BITS; ++pivot) {
        if (!has_pivot[pivot]) continue;
        result.rows.push_back(pivot_rows[pivot]);
        result.pivots.push_back(pivot);
    }
    return result;
}

cpp_int mask_coordinate_index(const Word5& mask, const XBasis& basis) {
    cpp_int index = 0;
    for (size_t coordinate = 0;
         coordinate < basis.pivots.size();
         ++coordinate) {
        if (word5_bit(mask, basis.pivots[coordinate])) {
            boost::multiprecision::bit_set(
                index,
                static_cast<unsigned>(coordinate)
            );
        }
    }
    return index;
}

string mask_expression(const Word5& mask, const Config& config) {
    ostringstream expression;
    bool first = true;
    for (int bit = 0; bit < STATE_BITS; ++bit) {
        if (!word5_bit(mask, bit)) continue;
        if (!first) expression << " + ";

        if (config.mode == "permutation") {
            expression << "x_" << bit;
        } else if (bit >= 64 && bit < 128) {
            expression << "k0_" << (bit - 64);
        } else if (bit >= 128 && bit < 192) {
            expression << "k1_" << (bit - 128);
        } else if (bit >= 192 && bit < 256) {
            expression
                << (config.mode == "equal" ? "n_" : "n0_")
                << (bit - 192);
        } else if (config.mode == "unrestricted"
                   && bit >= 256 && bit < 320) {
            expression << "n1_" << (bit - 256);
        } else {
            throw runtime_error(
                "x-mask contains an illegal bit for mode "
                + config.mode + ": " + to_string(bit)
            );
        }
        first = false;
    }
    return first ? "0" : expression.str();
}

void print_mask_bits(const Word5& mask) {
    cout << '[';
    for (int bit = 0; bit < STATE_BITS; ++bit) {
        if (bit != 0) cout << ", ";
        cout << (word5_bit(mask, bit) ? 1 : 0);
    }
    cout << ']';
}

int main(int argc, char** argv) {
    try {
        string manifest_path =
            argc > 1
                ? argv[1]
                : "output/record_trails_and_coefficients/step2_input_data.bin";

        auto manifest = load_manifest(manifest_path);
        const Config& config = manifest.first;
        const vector<DiffData>& diffs = manifest.second;

        cout << "ASCON v3 step2\n"
             << "  total_rounds                = " << config.total_rounds << '\n'
             << "  mode                        = " << config.mode << '\n'
             << "  IV                          = 0x"
             << hex << setw(16) << setfill('0') << config.iv
             << dec << setfill(' ') << '\n'
             << "  characteristic_model_number = "
             << config.characteristic_model_number << '\n'
             << "  trail_model_number          = "
             << config.trail_model_number << '\n'
             << "  characteristic_len          = "
             << config.characteristic_len() << '\n'
             << "  trail_len                   = "
             << config.trail_len() << '\n'
             << "  characteristics             = " << diffs.size() << '\n'
             << "  basis_number                = " << config.basis_number << '\n'
             << "  dim_truncated               = " << config.dim_truncated << '\n';

        vector<Word5> x_trails;
        vector<double> x_coefficients;
        unordered_map<Word5, size_t, Word5Hash> mask_to_index;
        map<double, uint64_t> all_correlation_counts;

        auto start = chrono::steady_clock::now();

        for (size_t diff_index = 0; diff_index < diffs.size(); ++diff_index) {
            const DiffData& diff = diffs[diff_index];
            vector<TrailRecord> records = load_basis_trails(
                diff.step1_path,
                config.dim_truncated,
                config.trail_len()
            );
            if (records.empty()) {
                throw runtime_error(
                    "no basis trails loaded for characteristic "
                    + to_string(diff_index)
                );
            }

            int dimension = static_cast<int>(records.size());
            if (dimension >= 63) {
                throw runtime_error("span dimension must be below 63");
            }
            uint64_t span_size = 1ULL << dimension;

            vector<Trail> basis_trails;
            basis_trails.reserve(records.size());
            for (auto& record : records) {
                basis_trails.push_back(move(record.trail));
            }

            map<double, uint64_t> dc_correlation_counts;
            uint64_t valid_count = 0;
            Trail current(static_cast<size_t>(config.trail_len()));

            for (uint64_t combo = 0; combo < span_size; ++combo) {
                clear_trail(current);
                for (int basis_index = 0;
                     basis_index < dimension;
                     ++basis_index) {
                    if (((combo >> basis_index) & 1ULL) == 0) continue;
                    for (int stage = 0;
                         stage < config.trail_len();
                         ++stage) {
                        for (int row = 0; row < STATE_ROWS; ++row) {
                            current[stage][row] ^=
                                basis_trails[basis_index][stage][row];
                        }
                    }
                }

                double correlation = quasidifferential_correlation(
                    config,
                    diff.characteristic,
                    current
                );
                correlation *= diff.cor_rb;

                ++dc_correlation_counts[correlation];
                ++all_correlation_counts[correlation];
                if (correlation == 0.0) continue;
                ++valid_count;

                const Word5& input_mask = current[0];
                auto found = mask_to_index.find(input_mask);
                if (found == mask_to_index.end()) {
                    size_t new_index = x_trails.size();
                    mask_to_index[input_mask] = new_index;
                    x_trails.push_back(input_mask);
                    x_coefficients.push_back(correlation);
                } else {
                    x_coefficients[found->second] += correlation;
                }
            }

            cout << "DC[" << diff_index << "]: loaded " << dimension
                 << " basis trails, spanned " << span_size
                 << ", valid " << valid_count
                 << ", accumulated masks " << x_trails.size() << '\n';
            print_correlation_distribution(
                dc_correlation_counts,
                "DC[" + to_string(diff_index)
                    + "] quasidifferential correlation distribution:"
            );
        }

        print_correlation_distribution(
            all_correlation_counts,
            "All quasidifferential correlation distribution:"
        );

        Word5 zero_mask = zero_word5();
        auto zero = mask_to_index.find(zero_mask);
        if (zero == mask_to_index.end()) {
            cout << "WARNING: C_[ux=0] not found\n";
        } else {
            double coefficient = x_coefficients[zero->second];
            cout << "Averaged-x differential = C_[ux=0] = 2^"
                 << log2(coefficient) << '\n';
        }

        cout << "\nGet mask-key bases:\n";
        XBasis x_basis = generate_x_basis(x_trails);
        size_t dim_x = x_basis.rows.size();
        cout << "dim x = " << dim_x << '\n';

        vector<string> basis_expressions;
        basis_expressions.reserve(dim_x);
        cout << "Basis expressions:\n";
        for (size_t i = 0; i < dim_x; ++i) {
            basis_expressions.push_back(
                mask_expression(x_basis.rows[i], config)
            );
            cout << "Basis[" << i << "] = ";
            print_mask_bits(x_basis.rows[i]);
            cout << " = " << basis_expressions.back() << '\n';
        }

        cout << "\nCompute basis contributions\n";
        vector<double> contributions(dim_x, 0.0);
        for (size_t i = 0; i < x_trails.size(); ++i) {
            for (size_t coordinate = 0; coordinate < dim_x; ++coordinate) {
                if (word5_bit(
                        x_trails[i],
                        x_basis.pivots[coordinate]
                    )) {
                    contributions[coordinate] += abs(x_coefficients[i]);
                }
            }
        }

        vector<size_t> ranked(dim_x);
        iota(ranked.begin(), ranked.end(), size_t{0});
        stable_sort(
            ranked.begin(),
            ranked.end(),
            [&contributions](size_t left, size_t right) {
                return contributions[left] > contributions[right];
            }
        );

        double total_contribution = accumulate(
            contributions.begin(),
            contributions.end(),
            0.0
        );
        cout << "Basis contributions (total="
             << scientific << setprecision(6)
             << total_contribution << "):\n";
        for (size_t rank = 0; rank < ranked.size(); ++rank) {
            size_t basis_index = ranked[rank];
            double percentage =
                total_contribution > 0.0
                    ? contributions[basis_index]
                        / total_contribution * 100.0
                    : 0.0;
            cout << "  rank[" << rank << "]: basis[" << basis_index
                 << "] = " << contributions[basis_index]
                 << " (" << fixed << setprecision(2)
                 << percentage << "%)\n";
        }
        cout << defaultfloat;

        cout << "\nCumulative contribution:\n";
        double cumulative = 0.0;
        for (size_t count = 1; count <= ranked.size(); ++count) {
            cumulative += contributions[ranked[count - 1]];
            double percentage =
                total_contribution > 0.0
                    ? cumulative / total_contribution * 100.0
                    : 0.0;
            cout << "  top " << setw(2) << count << ": "
                 << fixed << setprecision(1)
                 << percentage << "%\n";
        }
        cout << defaultfloat;

        filesystem::create_directories(
            "output/record_bases_and_contributions"
        );
        string save_path =
            "output/record_bases_and_contributions/"
            "step2_bases_contributions_r"
            + to_string(config.total_rounds)
            + "_" + config.d_str
            + "_mode_" + config.mode
            + "_basis_number_" + to_string(config.basis_number)
            + "_weight_" + to_string(config.weight_range)
            + "_dim_" + to_string(config.dim_truncated)
            + ".jsonl";

        ofstream output(save_path);
        if (!output) {
            throw runtime_error("cannot create output file: " + save_path);
        }

        output << "{\"mode\":\"" << config.mode
               << "\",\"dim_x\":" << dim_x << ",\"x_basis\":[";
        for (size_t basis_index = 0;
             basis_index < x_basis.rows.size();
             ++basis_index) {
            if (basis_index != 0) output << ',';
            output << '[';
            for (int bit = 0; bit < STATE_BITS; ++bit) {
                if (bit != 0) output << ',';
                output << (
                    word5_bit(x_basis.rows[basis_index], bit) ? 1 : 0
                );
            }
            output << ']';
        }

        output << "],\"contrib\":["
               << scientific << setprecision(17);
        for (size_t i = 0; i < contributions.size(); ++i) {
            if (i != 0) output << ',';
            output << contributions[i];
        }

        output << "],\"ranked\":[";
        for (size_t i = 0; i < ranked.size(); ++i) {
            if (i != 0) output << ',';
            output << ranked[i];
        }

        output << "],\"coord_list\":[";
        for (size_t i = 0; i < x_trails.size(); ++i) {
            if (i != 0) output << ',';
            cpp_int coordinate =
                mask_coordinate_index(x_trails[i], x_basis);
            output << '[' << coordinate << ','
                   << scientific << setprecision(17)
                   << x_coefficients[i] << ']';
        }
        output << "]}\n";

        cout << "\nSaved to: " << save_path << '\n';

        double seconds = chrono::duration<double>(
            chrono::steady_clock::now() - start
        ).count();
        cout << "Time: " << seconds << "s\n";
        return 0;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
