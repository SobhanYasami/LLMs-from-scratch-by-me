//! Build script. From Zig-version/:
//!
//!   zig build run     build and run the demo
//!   zig build test    run the tests in src/tokenization/

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "llm-from-scratch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Build and run the demo").dependOn(&run.step);

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tokenization/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.step("test", "Run the unit tests").dependOn(&b.addRunArtifact(tests).step);
}
