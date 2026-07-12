package com.iflytek.skillhub.domain.modelpolicy;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

public final class PolicyChecksum {
    private PolicyChecksum() {}

    public static String frame(Object... fields) {
        StringBuilder result = new StringBuilder();
        for (Object field : fields) {
            if (field instanceof List<?> list) {
                result.append('L').append(list.size()).append(':');
                list.forEach(value -> append(result, String.valueOf(value)));
            } else {
                append(result, String.valueOf(field));
            }
        }
        return result.toString();
    }

    public static String sha256(String canonical) {
        try {
            return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(canonical.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static void append(StringBuilder target, String value) {
        target.append('V').append(value.getBytes(StandardCharsets.UTF_8).length).append(':').append(value);
    }
}
