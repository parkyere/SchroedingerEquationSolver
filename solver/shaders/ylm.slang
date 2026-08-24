// Real Y_lm, l=0..5; shared by synth / mcwf_axpy / project_deposit.
module ylm;
import complex;

public static const float PI = kPi;

// r=0 singularity pinned by caller.
public float real_Ylm(int l, int m, float x, float y, float z) {
    float r2 = x * x + y * y + z * z;
    if (l == 0) return 1.0 / (2.0 * sqrt(PI));
    if (r2 == 0.0) return 0.0;
    switch (l) {
    case 1: {
        float c = sqrt(3.0 / (4.0 * PI)) / sqrt(r2);
        switch (m) {
        case -1: return c * y;
        case 0: return c * z;
        default: return c * x;
        }
    }
    case 2: {
        float c = sqrt(15.0 / PI) / r2;
        switch (m) {
        case -2: return 0.5 * c * x * y;
        case -1: return 0.5 * c * y * z;
        case 0: return 0.25 * sqrt(5.0 / PI) * (3.0 * z * z - r2) / r2;
        case 1: return 0.5 * c * z * x;
        default: return 0.25 * c * (x * x - y * y);
        }
    }
    case 3: {
        float r3 = r2 * sqrt(r2);
        switch (m) {
        case -3: return 0.25 * sqrt(35.0 / (2.0 * PI)) * y * (3.0 * x * x - y * y) / r3;
        case -2: return 0.5 * sqrt(105.0 / PI) * x * y * z / r3;
        case -1: return 0.25 * sqrt(21.0 / (2.0 * PI)) * y * (5.0 * z * z - r2) / r3;
        case 0: return 0.25 * sqrt(7.0 / PI) * z * (5.0 * z * z - 3.0 * r2) / r3;
        case 1: return 0.25 * sqrt(21.0 / (2.0 * PI)) * x * (5.0 * z * z - r2) / r3;
        case 2: return 0.25 * sqrt(105.0 / PI) * z * (x * x - y * y) / r3;
        default: return 0.25 * sqrt(35.0 / (2.0 * PI)) * x * (x * x - 3.0 * y * y) / r3;
        }
    }
    case 4: {
        float r4 = r2 * r2;
        switch (m) {
        case -4: return 0.75 * sqrt(35.0 / PI) * x * y * (x * x - y * y) / r4;
        case -3: return 0.75 * sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x * x - y * y) / r4;
        case -2: return 0.75 * sqrt(5.0 / PI) * x * y * (7.0 * z * z - r2) / r4;
        case -1: return 0.75 * sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
        case 0: return (3.0 / 16.0) * sqrt(1.0 / PI) *
                       (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
        case 1: return 0.75 * sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
        case 2: return 0.375 * sqrt(5.0 / PI) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
        case 3: return 0.75 * sqrt(35.0 / (2.0 * PI)) * x * z * (x * x - 3.0 * y * y) / r4;
        default: return (3.0 / 16.0) * sqrt(35.0 / PI) *
                        (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
        }
    }
    default: {  // l == 5 (caller guarantees l<=5)
        float r5 = r2 * r2 * sqrt(r2);
        float x2 = x * x;
        float y2 = y * y;
        float z2 = z * z;
        float zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;
        switch (m) {
        case -5: return (1.0 / 32.0) * sqrt(1386.0 / PI) *
                        y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
        case -4: return (1.0 / 16.0) * sqrt(3465.0 / PI) * 4.0 * x * y * (x2 - y2) * z / r5;
        case -3: return (1.0 / 32.0) * sqrt(770.0 / PI) * y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
        case -2: return (1.0 / 8.0) * sqrt(1155.0 / PI) * 2.0 * x * y * z * (3.0 * z2 - r2) / r5;
        case -1: return (1.0 / 16.0) * sqrt(165.0 / PI) * y * zpoly / r5;
        case 0: return (1.0 / 16.0) * sqrt(11.0 / PI) *
                       z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
        case 1: return (1.0 / 16.0) * sqrt(165.0 / PI) * x * zpoly / r5;
        case 2: return (1.0 / 8.0) * sqrt(1155.0 / PI) * (x2 - y2) * z * (3.0 * z2 - r2) / r5;
        case 3: return (1.0 / 32.0) * sqrt(770.0 / PI) * x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
        case 4: return (1.0 / 16.0) * sqrt(3465.0 / PI) *
                       (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
        default: return (1.0 / 32.0) * sqrt(1386.0 / PI) * x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
        }
    }
    }
}
