// Module 3.4 unit tests: select_experts() against each SelectionPolicy,
// per dev_spec.md Module 3.4's cache-replacement-policy simplification
// (see docs/PROGRESS.md's placeholder ledger -- this is what replaces it).
#include <gtest/gtest.h>

#include "selection_policy.h"

using namespace freetoken::core;

TEST(SelectExperts, FirstInOrderTakesFirstQUnchanged) {
    HostPoolConfig cfg;
    HostResidentPool pool(cfg);
    SelectionHistory history;

    std::vector<uint32_t> missing = {10, 20, 30, 40, 50};
    QStarSplit split = select_experts(SelectionPolicy::FirstInOrder, missing, 2, pool, history);

    EXPECT_EQ(split.fill_set, (std::vector<uint32_t>{10, 20}));
    EXPECT_EQ(split.compute_set, (std::vector<uint32_t>{30, 40, 50}));
}

TEST(SelectExperts, MostRecentlyRequestedPrioritizesRecentMisses) {
    HostPoolConfig cfg;
    HostResidentPool pool(cfg);
    SelectionHistory history;

    // Step 1: {1, 2, 3} all miss for the first time. No prior history, so
    // this call's ranking is a three-way tie -- order is unaffected. But
    // it DOES record tick=1 for all three, for future calls to use.
    select_experts(SelectionPolicy::MostRecentlyRequested, {1, 2, 3}, 3, pool, history);

    // Step 2: query {4, 1, 2}. Ranking uses history as it stood BEFORE
    // this call: expert 4 has no history (tick 0), experts 1 and 2 both
    // have tick=1 from step 1 (tied with each other, both beat expert 4).
    // Tie-break is input order, and 1 appears before 2 in {4, 1, 2} -- so
    // expert 1 wins the one fill slot.
    QStarSplit split = select_experts(SelectionPolicy::MostRecentlyRequested, {4, 1, 2}, 1, pool, history);
    EXPECT_EQ(split.fill_set, (std::vector<uint32_t>{1}));
    // (This call also just recorded tick=2 for 4, 1, AND 2 -- all three
    // were in ITS missing set, regardless of which one got the fill slot.)

    // Step 3: query {4, 1, 2} again. Now all three are tied at tick=2
    // (every one of them was in step 2's missing set). A three-way tie
    // falls back to input order -- expert 4 is first in {4, 1, 2}.
    QStarSplit split3 = select_experts(SelectionPolicy::MostRecentlyRequested, {4, 1, 2}, 1, pool, history);
    EXPECT_EQ(split3.fill_set, (std::vector<uint32_t>{4}));
}

TEST(SelectExperts, SmallestFirstPrioritizesSmallestExperts) {
    HostPoolConfig cfg;
    HostResidentPool pool(cfg);
    SelectionHistory history;

    char big[300] = {0};
    char small[10] = {0};
    char medium[100] = {0};
    pool.register_expert(1, big, sizeof(big));
    pool.register_expert(2, small, sizeof(small));
    pool.register_expert(3, medium, sizeof(medium));

    QStarSplit split = select_experts(SelectionPolicy::SmallestFirst, {1, 2, 3}, 2, pool, history);
    EXPECT_EQ(split.fill_set, (std::vector<uint32_t>{2, 3}));  // small, then medium
    EXPECT_EQ(split.compute_set, (std::vector<uint32_t>{1}));  // big left for compute
}
