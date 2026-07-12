package com.iflytek.skillhub.domain.modelpolicy;

import java.util.List;

/**
 * Temporary golden-vector generator. Uses the REAL PolicyChecksum.java from
 * SkillHub (copied verbatim) to produce authoritative framed strings + SHA-256
 * digests for fixed effective-policy field tuples. Output is JSON-lines so the
 * Hermes Python test suite can consume them as constants.
 *
 * Build/run:
 *   JAVA_HOME=/opt/homebrew/opt/openjdk@21
 *   $JAVA_HOME/bin/javac -d /tmp/policy-golden/out \
 *     /tmp/policy-golden/src/com/iflytek/skillhub/domain/modelpolicy/*.java
 *   $JAVA_HOME/bin/java -cp /tmp/policy-golden/out \
 *     com.iflytek.skillhub.domain.modelpolicy.GoldenVectorsMain
 */
public final class GoldenVectorsMain {
    private GoldenVectorsMain() {}

    record Vector(String id, String mode, List<String> allowedModels, String defaultModel,
                  List<String> fallbackModels, boolean localProviderAllowed) {}

    public static void main(String[] args) {
        List<Vector> vectors = List.of(
            new Vector("single_no_fallback_local_false",
                "ENTERPRISE_MANAGED",
                List.of("enterprise/deepseek-chat"),
                "enterprise/deepseek-chat",
                List.of(),
                false),
            new Vector("multi_allowed_ordered_fallback_local_false",
                "ENTERPRISE_MANAGED",
                List.of("enterprise/deepseek-chat", "enterprise/gpt-5"),
                "enterprise/deepseek-chat",
                List.of("enterprise/gpt-5"),
                false),
            new Vector("local_true",
                "USER_SELECTABLE",
                List.of("local/ollama-qwen", "openai/gpt-5"),
                "local/ollama-qwen",
                List.of("openai/gpt-5"),
                true),
            new Vector("non_ascii_utf8_byte_length",
                "ENTERPRISE_MANAGED",
                List.of("enterprise/中文模型", "enterprise/café"),
                "enterprise/中文模型",
                List.of("enterprise/café"),
                false),
            // Cross-check against the literal golden SHA asserted by SkillHub's
            // own EffectiveModelPolicyResolverTest.canonicalPolicyChecksumAndVersionAreStable.
            new Vector("java_test_cross_check",
                "USER_SELECTABLE",
                List.of("openai/gpt-5", "anthropic/claude-sonnet"),
                "anthropic/claude-sonnet",
                List.of(),
                false)
        );

        for (Vector v : vectors) {
            String framed = PolicyChecksum.frame(
                v.mode(), v.allowedModels(), v.defaultModel(),
                v.fallbackModels(), Boolean.toString(v.localProviderAllowed()));
            String sha = PolicyChecksum.sha256(framed);
            String policyVersion = "sha256:" + sha.substring(0, 12);
            System.out.println(toJson(v, framed, sha, policyVersion));
        }
    }

    private static String toJson(Vector v, String framed, String sha, String policyVersion) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"id\":\"").append(v.id()).append("\"");
        sb.append(",\"mode\":\"").append(v.mode()).append("\"");
        sb.append(",\"allowedModels\":").append(jsonStrList(v.allowedModels()));
        sb.append(",\"defaultModel\":\"").append(v.defaultModel()).append("\"");
        sb.append(",\"fallbackModels\":").append(jsonStrList(v.fallbackModels()));
        sb.append(",\"localProviderAllowed\":").append(v.localProviderAllowed());
        sb.append(",\"framed\":").append(jsonStr(framed));
        sb.append(",\"policySha256\":\"").append(sha).append("\"");
        sb.append(",\"policyVersion\":\"").append(policyVersion).append("\"");
        sb.append("}");
        return sb.toString();
    }

    private static String jsonStrList(List<String> items) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(jsonStr(items.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String jsonStr(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append("\"");
        return sb.toString();
    }
}
