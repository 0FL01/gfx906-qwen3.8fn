#include "gdn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t checks = 0;
std::size_t rejections = 0;
double max_output_error = 0;
double max_state_error = 0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

// Fixed BEFORE measurement: FP32 vs independent double oracle; no tolerance
// tuning on observed errors. Snapshot/chunk parity is tested exactly instead.
void near(float actual, double expected, bool state = false) {
    const double error = std::abs(double(actual) - expected);
    if (state) max_state_error = std::max(max_state_error, error);
    else max_output_error = std::max(max_output_error, error);
    check(std::isfinite(actual) && std::isfinite(expected) &&
          error <= (state ? 1.0e-6 : 2.0e-6) + (state ? 3.0e-6 : 1.0e-5) * std::abs(expected),
          state ? "independent recurrent state mismatch" : "independent output mismatch");
}

template<class Error = std::invalid_argument, class Function>
void rejected(Function function) {
    try { function(); }
    catch (const Error&) { ++rejections; return; }
    throw std::runtime_error("invalid/overflowing GDN operation accepted");
}

// Test data is created in authoritative HF GROUPED value-head order. Pack only
// by the converter's mathematical permutation; production sees tiled arrays.
// This makes the oracle's h/ratio head association independent of GdnCpu's h%K.
struct Fixture {
    qwen::GdnConfig c;
    std::size_t tokens;
    std::vector<float> conv, dt, a, norm, qkv, z, alpha, beta;
    std::vector<float> packed_conv, packed_dt, packed_a, packed_qkv, packed_z,
                       packed_alpha, packed_beta;

    std::size_t tiled(std::size_t hf_head) const {
        const auto ratio = c.value_heads / c.key_heads;
        return (hf_head % ratio) * c.key_heads + hf_head / ratio;
    }
    Fixture(qwen::GdnConfig config, std::size_t n)
        : c(config), tokens(n), conv(c.qkv_elements() * c.conv_width), dt(c.value_heads),
          a(c.value_heads), norm(c.value_head_dim), qkv(n * c.qkv_elements()),
          z(n * c.value_elements()), alpha(n * c.value_heads), beta(n * c.value_heads) {
        const auto keys = c.key_elements();
        for (std::size_t f = 0; f < c.qkv_elements(); ++f)
            for (std::size_t w = 0; w < c.conv_width; ++w)
                conv[f * c.conv_width + w] = float(0.2 * std::sin(double(f + 3 * w) * 0.37) +
                                                 0.1 * double(w + 1));
        for (std::size_t h = 0; h < c.value_heads; ++h) {
            dt[h] = float(0.19 * std::cos(double(h) * 0.4));
            a[h] = float(-std::exp(-1.1 + 0.015 * double(h)));
        }
        for (std::size_t v = 0; v < c.value_head_dim; ++v)
            norm[v] = float(0.7 + 0.4 * std::cos(double(v) * 0.5));
        for (std::size_t t = 0; t < n; ++t) {
            for (std::size_t f = 0; f < c.qkv_elements(); ++f)
                qkv[t * c.qkv_elements() + f] = float(0.8 * std::sin(double(f + 17 * t) * 0.17) +
                                                     0.3 * std::cos(double(3 * f + t) * 0.09));
            for (std::size_t i = 0; i < c.value_elements(); ++i)
                z[t * c.value_elements() + i] = float(1.4 * std::sin(double(i + 7 * t) * 0.39));
            for (std::size_t h = 0; h < c.value_heads; ++h) {
                alpha[t * c.value_heads + h] = float(0.6 * std::cos(double(h + 11 * t) * 0.41));
                beta[t * c.value_heads + h] = float(1.8 * std::sin(double(h + 5 * t) * 0.28));
            }
        }
        packed_conv = conv;
        packed_qkv = qkv;
        packed_dt.resize(dt.size()); packed_a.resize(a.size()); packed_z.resize(z.size());
        packed_alpha.resize(alpha.size()); packed_beta.resize(beta.size());
        for (std::size_t h = 0; h < c.value_heads; ++h) {
            const auto th = tiled(h);
            packed_dt[th] = dt[h]; packed_a[th] = a[h];
            for (std::size_t v = 0; v < c.value_head_dim; ++v) {
                const auto f = 2 * keys + h * c.value_head_dim + v;
                const auto tf = 2 * keys + th * c.value_head_dim + v;
                for (std::size_t w = 0; w < c.conv_width; ++w)
                    packed_conv[tf * c.conv_width + w] = conv[f * c.conv_width + w];
                for (std::size_t t = 0; t < n; ++t) {
                    packed_qkv[t * c.qkv_elements() + tf] = qkv[t * c.qkv_elements() + f];
                    packed_z[t * c.value_elements() + th * c.value_head_dim + v] =
                        z[t * c.value_elements() + h * c.value_head_dim + v];
                }
            }
            for (std::size_t t = 0; t < n; ++t) {
                packed_alpha[t * c.value_heads + th] = alpha[t * c.value_heads + h];
                packed_beta[t * c.value_heads + th] = beta[t * c.value_heads + h];
            }
        }
    }
    qwen::GdnParameters parameters() const {
        return {packed_conv, packed_dt, packed_a, norm};
    }
    qwen::GdnInput input(std::size_t start, std::size_t n = 1) const {
        return {std::span(packed_qkv).subspan(start * c.qkv_elements(), n * c.qkv_elements()),
                std::span(packed_z).subspan(start * c.value_elements(), n * c.value_elements()),
                std::span(packed_alpha).subspan(start * c.value_heads, n * c.value_heads),
                std::span(packed_beta).subspan(start * c.value_heads, n * c.value_heads)};
    }
};

double logistic(double x) { return 0.5 * (1.0 + std::tanh(0.5 * x)); }
double softplus(double x) { return std::max(0.0, x) + std::log1p(std::exp(-std::abs(x))); }

// Independent oracle: double, HF grouped heads, K-major state, input history
// gathered by absolute token ID (no mutable channel history buffer). For small
// shapes explicitly multiply decay*(I-beta*k*k^T) by the old dense matrix,
// rather than mirroring the production prediction/delta/column update loop.
struct Oracle {
    const Fixture& f;
    std::vector<double> state;
    std::vector<std::size_t> seen;
    explicit Oracle(const Fixture& fixture) : f(fixture), state(f.c.recurrent_elements(), 0) {}

    std::vector<double> step(std::size_t token) {
        const auto& c = f.c;
        const auto keys = c.key_elements();
        std::vector<double> x(c.qkv_elements()), result(c.value_elements());
        seen.push_back(token);
        for (std::size_t feature = 0; feature < x.size(); ++feature) {
            double convolved = 0;
            for (std::size_t w = 0; w < c.conv_width; ++w) {
                const auto lag = c.conv_width - 1 - w;
                if (lag < seen.size())
                    convolved += f.qkv[seen[seen.size() - 1 - lag] * x.size() + feature] *
                                 double(f.conv[feature * c.conv_width + w]);
            }
            x[feature] = convolved * logistic(convolved);
        }
        for (std::size_t h = 0; h < 2 * c.key_heads; ++h) {
            double squared = 1.0e-6;
            for (std::size_t k = 0; k < c.key_head_dim; ++k)
                squared += x[h * c.key_head_dim + k] * x[h * c.key_head_dim + k];
            const double norm = std::sqrt(squared) * (h < c.key_heads ? std::sqrt(double(c.key_head_dim)) : 1.0);
            for (std::size_t k = 0; k < c.key_head_dim; ++k) x[h * c.key_head_dim + k] /= norm;
        }
        std::vector<double> next(state.size());
        const auto ratio = c.value_heads / c.key_heads;
        for (std::size_t h = 0; h < c.value_heads; ++h) {
            const auto kh = h / ratio;
            const double beta = logistic(f.beta[token * c.value_heads + h]);
            const double decay = std::exp(double(f.a[h]) * softplus(
                double(f.alpha[token * c.value_heads + h]) + f.dt[h]));
            const auto base = h * c.key_head_dim * c.value_head_dim;
            for (std::size_t v = 0; v < c.value_head_dim; ++v) {
                // For real geometry the algebraically identical rank-one form
                // avoids a D^3 oracle cost. Tiny fixtures use the dense matrix.
                double prediction = 0;
                if (c.key_head_dim > 8)
                    for (std::size_t j = 0; j < c.key_head_dim; ++j)
                        prediction += x[keys + kh * c.key_head_dim + j] * state[base + j * c.value_head_dim + v];
                for (std::size_t k = 0; k < c.key_head_dim; ++k) {
                    const double kk = x[keys + kh * c.key_head_dim + k];
                    double retained = 0;
                    if (c.key_head_dim <= 8) {
                        for (std::size_t j = 0; j < c.key_head_dim; ++j) {
                            const double kj = x[keys + kh * c.key_head_dim + j];
                            const double transition = decay * ((k == j ? 1.0 : 0.0) - beta * kk * kj);
                            retained += transition * state[base + j * c.value_head_dim + v];
                        }
                    } else {
                        retained = decay * (state[base + k * c.value_head_dim + v] - beta * kk * prediction);
                    }
                    const auto i = base + k * c.value_head_dim + v;
                    next[i] = retained + beta * kk * x[2 * keys + h * c.value_head_dim + v];
                    result[h * c.value_head_dim + v] += x[kh * c.key_head_dim + k] * next[i];
                }
            }
            double variance = 0;
            for (std::size_t v = 0; v < c.value_head_dim; ++v)
                variance += std::pow(result[h * c.value_head_dim + v], 2);
            const double denominator = std::sqrt(variance / double(c.value_head_dim) + c.rms_epsilon);
            for (std::size_t v = 0; v < c.value_head_dim; ++v)
                result[h * c.value_head_dim + v] = result[h * c.value_head_dim + v] / denominator *
                    f.norm[v] * logistic(f.z[token * c.value_elements() + h * c.value_head_dim + v]);
        }
        state = std::move(next);
        return result;
    }
};

void compare_oracle(const Fixture& f, const Oracle& oracle, const qwen::GdnCpu& cpu,
                    std::span<const float> output, std::span<const double> expected) {
    const auto& c = f.c;
    for (std::size_t h = 0; h < c.value_heads; ++h) {
        const auto th = f.tiled(h);
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            near(output[th * c.value_head_dim + v], expected[h * c.value_head_dim + v]);
            for (std::size_t k = 0; k < c.key_head_dim; ++k)
                near(cpu.state().recurrent()[(th * c.value_head_dim + v) * c.key_head_dim + k],
                     oracle.state[(h * c.key_head_dim + k) * c.value_head_dim + v], true);
        }
    }
    for (std::size_t feature = 0; feature < c.qkv_elements(); ++feature)
        for (std::size_t age = 0; age + 1 < c.conv_width; ++age) {
            const auto lag = c.conv_width - 1 - age;
            float expected_history = 0;
            if (lag <= oracle.seen.size())
                expected_history = f.packed_qkv[oracle.seen[oracle.seen.size() - lag] * c.qkv_elements() + feature];
            check(cpu.state().conv_history()[feature * (c.conv_width - 1) + age] == expected_history,
                  "raw causal conv history mismatch");
        }
    check(cpu.state().consumed_tokens() == oracle.seen.size(), "consumed input count mismatch");
}

void same_state(const qwen::GdnCheckpoint& a, const qwen::GdnCheckpoint& b) {
    check(a.consumed_tokens() == b.consumed_tokens(), "checkpoint consumed count mismatch");
    check(std::equal(a.recurrent().begin(), a.recurrent().end(), b.recurrent().begin(), b.recurrent().end()),
          "checkpoint recurrent mismatch");
    check(std::equal(a.conv_history().begin(), a.conv_history().end(), b.conv_history().begin(), b.conv_history().end()),
          "checkpoint convolution mismatch");
}

void oracle_test(qwen::GdnConfig c, std::size_t tokens) {
    Fixture f(c, tokens);
    qwen::GdnCpu cpu(c, f.parameters());
    Oracle oracle(f);
    std::vector<float> out(c.value_elements());
    for (std::size_t t = 0; t < tokens; ++t) {
        cpu.step(f.input(t), out);
        const auto expected = oracle.step(t);
        compare_oracle(f, oracle, cpu, out, expected);
    }
    cpu.reset();
    qwen::GdnCheckpoint zero(c);
    same_state(cpu.state(), zero);
}

void chunk_test() {
    for (auto width : {std::size_t(1), std::size_t(4)}) {
        qwen::GdnConfig c{2, 6, 3, 5, width, 2.0e-5f}; // rectangular state exposes transposes
        Fixture f(c, 19);
        qwen::GdnCpu seq(c, f.parameters()), whole(c, f.parameters()), split(c, f.parameters());
        std::vector<float> expected(f.tokens * c.value_elements()), all(expected.size()), parts(expected.size());
        for (std::size_t t = 0; t < f.tokens; ++t)
            seq.step(f.input(t), std::span(expected).subspan(t * c.value_elements(), c.value_elements()));
        whole.run(f.input(0, f.tokens), f.tokens, all);
        for (std::size_t start = 0, n = 1; start < f.tokens; n = n % 5 + 1) {
            const auto count = std::min(n, f.tokens - start);
            split.run(f.input(start, count), count, std::span(parts).subspan(start * c.value_elements(), count * c.value_elements()));
            start += count;
        }
        check(all == expected && parts == expected, "sequence/chunk boundary output mismatch");
        same_state(seq.state(), whole.state()); same_state(seq.state(), split.state());
        qwen::GdnCheckpoint before(c), empty(c);
        split.save(before);
        split.run({}, 0, {}, std::span(&empty, 1));
        same_state(before, empty); same_state(before, split.state());
    }
}

void checkpoint_test() {
    qwen::GdnConfig c{2, 6, 3, 4, 4, 1.0e-6f};
    Fixture f(c, 28);
    qwen::GdnCpu cpu(c, f.parameters());
    std::vector<float> out(c.value_elements()), verify(3 * c.value_elements());
    std::vector<std::size_t> committed{0, 1, 2, 3, 4};
    for (auto t : committed) cpu.step(f.input(t), out);
    std::array<qwen::GdnCheckpoint, 4> prefix{
        qwen::GdnCheckpoint(c), qwen::GdnCheckpoint(c), qwen::GdnCheckpoint(c), qwen::GdnCheckpoint(c)};
    const float* const active_address = cpu.state().recurrent().data();
    const float* const history_address = cpu.state().conv_history().data();
    const float* const snapshot_address = prefix[0].recurrent().data();
    for (std::size_t window = 0; window < 3; ++window) {
        const auto pending = 5 + 6 * window;
        cpu.run(f.input(pending, 3), 3, verify, prefix);
        // Check the baseline prefix as well as accept0/1/2. Verify consumes one
        // pending target input before two drafts: accept a means prefix[1+a].
        for (std::size_t consumed = 0; consumed <= 3; ++consumed) {
            cpu.restore(prefix[consumed]);
            qwen::GdnCpu replay(c, f.parameters());
            Oracle oracle(f);
            for (auto t : committed) { replay.step(f.input(t), out); (void)oracle.step(t); }
            for (std::size_t i = 0; i < consumed; ++i) {
                replay.step(f.input(pending + i), out); (void)oracle.step(pending + i);
            }
            same_state(cpu.state(), replay.state());
            cpu.step(f.input(pending + 4), out); // distinct replacement/continuation input
            const auto expected = oracle.step(pending + 4);
            compare_oracle(f, oracle, cpu, out, expected);
            std::vector<float> replay_out(c.value_elements());
            replay.step(f.input(pending + 4), replay_out);
            check(out == replay_out, "continuation after rejection mismatch");
            same_state(cpu.state(), replay.state());
        }
        // Keep accepting zero drafts across windows; rejected histories must not
        // leak into the next window even when the convolution has a full tail.
        cpu.restore(prefix[1]); cpu.step(f.input(pending + 4), out);
        committed.push_back(pending); committed.push_back(pending + 4);
    }
    check(cpu.state().recurrent().data() == active_address && cpu.state().conv_history().data() == history_address &&
          prefix[0].recurrent().data() == snapshot_address, "persistent buffers reallocated");
}

void controls_test() {
    const qwen::GdnConfig c{1, 1, 1, 1, 1, 1.0e-6f};
    const std::array<float, 3> conv{1, 1, 1};
    const std::array<float, 1> dt{0.75f}, a{-0.4f}, norm{2.0f};
    qwen::GdnCpu cpu(c, {conv, dt, a, norm});
    const std::array<float, 3> qkv{0.01f, 0.02f, 0.8f};
    std::array<float, 1> z{0}, alpha{-0.2f}, beta{-1.3f}, out{};
    double state = 0;
    for (auto b : {-1.3f, 100.0f, -100.0f}) {
        beta[0] = b;
        cpu.step({qkv, z, alpha, beta}, out);
        const double q0 = qkv[0] * logistic(qkv[0]);
        const double k0 = qkv[1] * logistic(qkv[1]);
        const double q = q0 / std::sqrt(q0 * q0 + 1.0e-6);
        const double k = k0 / std::sqrt(k0 * k0 + 1.0e-6);
        const double v = qkv[2] * logistic(qkv[2]);
        const double decay = std::exp(a[0] * softplus(double(alpha[0]) + dt[0]));
        state = decay * (1.0 - logistic(b) * k * k) * state + logistic(b) * k * v;
        near(cpu.state().recurrent()[0], state, true);
        const double read = q * state;
        near(out[0], read / std::sqrt(read * read + c.rms_epsilon)); // weight2 * sigmoid(0)=1; NOT SiLU(0)=0
        check(out[0] > 0, "output gate used SiLU instead of sigmoid");
    }
    // Extreme finite sigmoid/softplus logits must work without exp overflow.
    for (auto extreme : {-100.0f, 100.0f}) {
        alpha[0] = extreme; z[0] = extreme; beta[0] = extreme;
        cpu.step({qkv, z, alpha, beta}, out);
        check(std::isfinite(out[0]), "extreme finite control produced nonfinite output");
    }
    cpu.reset();
    const std::array<float, 3> zeros{};
    cpu.step({zeros, z, alpha, beta}, out);
    check(out[0] == 0 && cpu.state().recurrent()[0] == 0, "zero Q/K normalization invalid");
}

void rejection_test() {
    const qwen::GdnConfig c{2, 6, 3, 4, 4, 1.0e-6f};
    Fixture f(c, 3);
    qwen::GdnCpu cpu(c, f.parameters());
    std::vector<float> out(c.value_elements(), 17);
    qwen::GdnCheckpoint initial(c);
    auto invalid = c;
    invalid.key_heads = 0; rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    invalid = c; invalid.value_heads = 5; rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    invalid = c; invalid.key_head_dim = 0; rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    invalid = c; invalid.value_head_dim = 0; rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    invalid = c; invalid.conv_width = 0; rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    for (float eps : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        invalid = c; invalid.rms_epsilon = eps;
        rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    }
    invalid = c; invalid.key_head_dim = std::numeric_limits<std::size_t>::max();
    rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    invalid = c; invalid.conv_width = std::numeric_limits<std::size_t>::max();
    rejected([&] { (void)qwen::GdnCheckpoint(invalid); });
    for (std::size_t which = 0; which < 4; ++which) {
        auto p = f.parameters();
        std::array<std::span<const float>*, 4> spans{&p.conv_weights, &p.dt_bias, &p.ssm_a, &p.norm_weight};
        *spans[which] = spans[which]->first(spans[which]->size() - 1);
        rejected([&] { (void)qwen::GdnCpu(c, p); });
    }
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        for (auto* buffer : {&f.packed_conv, &f.packed_dt, &f.packed_a, &f.norm}) {
            const float previous = (*buffer)[0]; (*buffer)[0] = bad;
            rejected([&] { (void)qwen::GdnCpu(c, f.parameters()); });
            (*buffer)[0] = previous;
        }
        for (auto* buffer : {&f.packed_qkv, &f.packed_z, &f.packed_alpha, &f.packed_beta}) {
            const float previous = buffer->back(); buffer->back() = bad;
            std::vector<float> batch(3 * c.value_elements(), 17);
            rejected([&] { cpu.run(f.input(0, 3), 3, batch); });
            check(std::all_of(batch.begin(), batch.end(), [](float x) { return x == 17; }), "invalid chunk wrote output");
            same_state(cpu.state(), initial);
            buffer->back() = previous;
        }
    }
    const auto previous_a = f.packed_a[0]; f.packed_a[0] = 0.1f;
    rejected([&] { (void)qwen::GdnCpu(c, f.parameters()); }); f.packed_a[0] = previous_a;
    for (std::size_t which = 0; which < 4; ++which) {
        auto in = f.input(0);
        std::array<std::span<const float>*, 4> spans{&in.qkv, &in.z, &in.alpha, &in.beta};
        *spans[which] = spans[which]->first(spans[which]->size() - 1);
        rejected([&] { cpu.step(in, out); });
    }
    rejected([&] { cpu.step(f.input(0), std::span(out).first(out.size() - 1)); });
    rejected([&] { cpu.step(f.input(0), std::span(f.packed_z).first(c.value_elements())); });
    rejected([&] { cpu.run({}, std::numeric_limits<std::size_t>::max(), {}); });
    qwen::GdnCheckpoint wrong({1, 1, 2, 2, 1, 1.0e-6f});
    rejected([&] { cpu.save(wrong); }); rejected([&] { cpu.restore(wrong); });
    rejected([&] { cpu.run(f.input(0), 1, out, std::span(&initial, 1)); });
    // A caller may inspect checkpoint spans but cannot import poisoned state.
    qwen::GdnCheckpoint poisoned(c);
    const_cast<float*>(poisoned.recurrent().data())[0] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { cpu.restore(poisoned); });
    poisoned = qwen::GdnCheckpoint(c);
    const_cast<float*>(poisoned.conv_history().data())[0] = std::numeric_limits<float>::infinity();
    rejected([&] { cpu.restore(poisoned); });
    same_state(cpu.state(), initial);
    check(std::all_of(out.begin(), out.end(), [](float x) { return x == 17; }), "invalid step wrote output");

    // Finite input overflow is a runtime error and leaves this token atomic.
    auto values = f.packed_qkv;
    std::fill(f.packed_qkv.begin(), f.packed_qkv.end(), std::numeric_limits<float>::max());
    rejected<std::runtime_error>([&] { cpu.step(f.input(0), out); });
    same_state(cpu.state(), initial);
    check(std::all_of(out.begin(), out.end(), [](float x) { return x == 17; }), "overflow wrote output");
    f.packed_qkv = values;
    const float prev_alpha = f.packed_alpha[0];
    f.packed_alpha[0] = std::numeric_limits<float>::max();
    auto params = f.parameters();
    f.packed_dt[0] = std::numeric_limits<float>::max();
    qwen::GdnCpu controls_overflow(c, params);
    rejected<std::runtime_error>([&] { controls_overflow.step(f.input(0), out); });
    same_state(controls_overflow.state(), initial);
    f.packed_alpha[0] = prev_alpha;
    // Recovery after an arithmetic exception must not use dirty scratch.
    cpu.step(f.input(0), out);
    Oracle oracle(f);
    // Restore fixture dt (the first runtime copied its original parameters).
    f.packed_dt[0] = f.dt[0];
    const auto expected = oracle.step(0);
    compare_oracle(f, oracle, cpu, out, expected);
}

}

int main() {
    try {
        oracle_test({2, 6, 3, 5, 4, 2.0e-5f}, 13);
        oracle_test({1, 1, 4, 3, 1, 1.0e-6f}, 7);
        oracle_test({}, 4); // real QK16/V48/D128/conv4, every state element checked
        chunk_test(); checkpoint_test(); controls_test(); rejection_test();
        std::cout << "{\"test\":\"gdn\",\"passed\":true,\"checks\":" << checks
                  << ",\"rejections\":" << rejections << ",\"max_output_error\":" << max_output_error
                  << ",\"max_state_error\":" << max_state_error << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gdn test failed: " << error.what() << '\n';
        return 1;
    }
}
