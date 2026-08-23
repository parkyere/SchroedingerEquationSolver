// Scheduler contract: arcs' periodic callbacks cancel THEMSELVES and register
// new entries mid-call, so the running callable must never be destroyed under
// its own feet (self-cancel = deferred, captures alive through the call).

#include <gtest/gtest.h>

#include <memory>
import ses.scenario.scheduler;

namespace {

// Flips *flag when the last owner dies (tracks the callable's destruction).
struct DtorFlag {
    bool* flag;
    explicit DtorFlag(bool* f) : flag(f) {}
    ~DtorFlag() { *flag = true; }
};

TEST(Scheduler, SelfCancelKeepsTheRunningCallableAlive) {
    ses_shell::Scheduler s;
    bool destroyed = false;
    bool destroyed_seen_inside = false;
    bool ran = false;
    int id = 0;
    auto probe = std::make_shared<DtorFlag>(&destroyed);
    id = s.every(10, [&s, &id, probe, pd = &destroyed,
                      ps = &destroyed_seen_inside, pr = &ran] {
        // Locals FIRST: nothing below may read the captures after cancel.
        ses_shell::Scheduler* sched = &s;
        const int my_id = id;
        bool* seen = ps;
        bool* dest = pd;
        bool* ranp = pr;
        sched->cancel(my_id);
        *seen = *dest;  // callable (and its probe ref) must still be alive
        *ranp = true;
    });
    probe.reset();  // the callable now holds the last probe reference
    s.poll(10);
    EXPECT_TRUE(ran);
    EXPECT_FALSE(destroyed_seen_inside);
    EXPECT_TRUE(destroyed);  // released once the call completed
}

TEST(Scheduler, SelfCancelStopsRefiring) {
    ses_shell::Scheduler s;
    int fires = 0;
    int id = 0;
    id = s.every(10, [&s, &id, &fires] {
        ++fires;
        s.cancel(id);
    });
    s.poll(10);
    s.poll(20);
    s.poll(30);
    EXPECT_EQ(fires, 1);
}

TEST(Scheduler, PeriodicRefiresUntilCancelled) {
    ses_shell::Scheduler s;
    int fires = 0;
    const int id = s.every(10, [&fires] { ++fires; });
    s.poll(10);
    s.poll(20);
    s.poll(25);  // before due: no fire
    s.poll(30);
    EXPECT_EQ(fires, 3);
    s.cancel(id);
    s.poll(40);
    EXPECT_EQ(fires, 3);
}

TEST(Scheduler, EntriesRegisteredMidCallFireOnLaterPolls) {
    ses_shell::Scheduler s;
    int later = 0;
    int fires = 0;
    int id = 0;
    id = s.every(10, [&s, &id, &later, &fires] {
        // Locals first; the many push_backs may reallocate entries_.
        ses_shell::Scheduler* sched = &s;
        const int my_id = id;
        int* fp = &fires;
        int* lp = &later;
        for (int k = 0; k < 64; ++k) {
            sched->after(5, [lp] { ++*lp; });
        }
        sched->cancel(my_id);
        ++*fp;
    });
    s.poll(10);
    EXPECT_EQ(fires, 1);
    EXPECT_EQ(later, 0);  // new entries run on a later poll, not this one
    s.poll(16);
    EXPECT_EQ(later, 64);
}

}  // namespace
