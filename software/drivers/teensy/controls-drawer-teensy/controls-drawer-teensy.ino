#include <Encoder.h>

int limit_x_low = 9;
int limit_x_high = 10;
int limit_y_low = 11;
int limit_y_high = 12;

Encoder smallEnc(2, 3);
Encoder largeEnc(4, 5);

void setup() {
    // Start serial communication
    Serial.begin(115200);
}

long smallEncPos  = -999;
long largeEncPos  = -999;

void printState(int x_low, int x_high, int y_low, int y_high,
                int smallEncPos, int largeEncPos) {
    Serial.print(x_low);
    Serial.print(",");
    Serial.print(x_high);
    Serial.print(",");
    Serial.print(y_low);
    Serial.print(",");
    Serial.print(y_high);
    Serial.print(",");
    Serial.print(smallEncPos);
    Serial.print(",");
    Serial.print(largeEncPos);
    Serial.print("\n");
}

void loop() {
    delay(500);

    printState(digitalRead(limit_x_low),
               digitalRead(limit_x_high),
               digitalRead(limit_y_low),
               digitalRead(limit_y_high),
               smallEnc.read(),
               largeEnc.read());
}
