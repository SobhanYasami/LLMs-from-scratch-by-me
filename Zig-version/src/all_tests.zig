//! Test root: pulls in the tests of every module (`zig build test`).
//! It sits in src/ so attention can import ../tokenization/rng.zig.

test {
    _ = @import("tokenization/root.zig");
    _ = @import("attention/root.zig");
}
