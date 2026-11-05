import { describe, expect, it } from "vitest";
import { modelLabel } from "./DemoBadge";

describe("modelLabel", () => {
  it("formats Claude ids", () => expect(modelLabel("claude-haiku-5-5")).toBe("Claude Haiku 5.5"));
  it("keeps the fake and unknown ids", () => {
    expect(modelLabel("fake")).toBe("FakeLLM");
    expect(modelLabel("foo")).toBe("foo");
  });
});
