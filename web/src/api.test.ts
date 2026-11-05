import { describe, expect, it } from "vitest";
import { parseSse } from "./api";

describe("parseSse", () => {
  it("reads event name and JSON data", () => {
    expect(parseSse('event: delta\ndata: {"text": "Hallo"}')).toEqual(["delta", { text: "Hallo" }]);
  });
  it("ignores blocks without data", () => {
    expect(parseSse(": keep-alive")).toBeNull();
  });
});
