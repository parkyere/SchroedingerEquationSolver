module;
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
export module ses.scenario.scheduler;


// Single-threaded: all methods run on the SDL loop thread (no locking).
// CONTRACT: tests/scheduler_test.cpp -- a callback may cancel ITSELF and
// register new entries mid-call; the running callable must stay alive.


export namespace ses_shell {

class Scheduler {
public:
    void after(std::uint64_t delay_ms, std::function<void()> fn) {
        entries_.push_back({next_id_++, now_ + delay_ms, 0, std::move(fn)});
    }

    int every(std::uint64_t period_ms, std::function<void()> fn) {
        const int id = next_id_++;
        entries_.push_back({id, now_ + period_ms, period_ms, std::move(fn)});
        return id;
    }

    void cancel(int id) {
        for (Entry& e : entries_) {
            if (e.id == id) {
                e.fn = nullptr;    // reaped on the next poll
                e.period_ms = 0;   // and never re-armed mid-call
            }
        }
    }

    void poll(std::uint64_t now_ms) {
        now_ = now_ms;
        // Cached n: callbacks may push_back; those entries run next poll, not now.
        const std::size_t n = entries_.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (!entries_[i].fn || entries_[i].due > now_ms) {
                continue;
            }
            // Move the callable out BEFORE invoking: the callback may cancel
            // its own id or push_back (reallocating entries_) mid-call.
            const int id = entries_[i].id;
            const std::uint64_t period = entries_[i].period_ms;
            auto fn = std::move(entries_[i].fn);
            entries_[i].fn = nullptr;
            fn();
            if (period > 0) {
                // Re-arm by id lookup (the vector may have moved); a cancel
                // during the call zeroed period_ms, leaving the entry reaped.
                for (Entry& e : entries_) {
                    if (e.id == id && e.period_ms > 0) {
                        e.due = now_ms + period;
                        e.fn = std::move(fn);
                        break;
                    }
                }
            }
        }
        std::erase_if(entries_, [](const Entry& e) { return !e.fn; });
    }

private:
    struct Entry {
        int id;
        std::uint64_t due;
        std::uint64_t period_ms;  // 0 = one-shot
        std::function<void()> fn;
    };
    std::vector<Entry> entries_;
    std::uint64_t now_ = 0;
    int next_id_ = 1;
};

}  // namespace ses_shell
