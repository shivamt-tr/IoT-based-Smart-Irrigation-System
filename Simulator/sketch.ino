/* Include necessary libraries */
#include<DHT.h>
#include<Servo.h>
#include <LiquidCrystal_I2C.h>

/* Define pin and object for LDR sensor */
#define LDR_PIN 10

// LDR Characteristics
const float GAMMA = 0.7;
const float RL10 = 50;

LiquidCrystal_I2C lcd(0x27, 20, 4);

/* Define constants and variables for DHT and initialize objects */
#define DHTPIN_1 2
#define DHTPIN_2 4
#define DHTPIN_3 6
#define DHTPIN_4 8
#define DHTTYPE DHT22   // DHT 22  (AM2302)

DHT dht_1(DHTPIN_1, DHTTYPE);
DHT dht_2(DHTPIN_2, DHTTYPE);
DHT dht_3(DHTPIN_3, DHTTYPE);
DHT dht_4(DHTPIN_4, DHTTYPE);

float hum_1, hum_2, hum_3, hum_4;  // variables to store humidity values
float temp_1, temp_2, temp_3, temp_4;  // variables to store temperature values

/* Define constants and variables for Servo and initialize the object */
Servo servo_1, servo_2, servo_3, servo_4;
int angle_1 = 0, angle_2 = 0, angle_3 = 0, angle_4 = 0;

/* Define other variables */
int wp_1 = 0, wp_2 = 0, wp_3 = 0, wp_4 = 0;

/* Define min and max of temperature and humidity (calculated from the data) */
float max_temp = 100.0, min_temp = 20.0;
float max_hum = 51.867867867867865, min_hum = 8.132132132132131;

// Define some values for neural net architecture
#define DIMENSION 2
#define LAYER1 16
#define LAYER2 8
#define OUT_LAYER 8

// Define the weights with values from pre-trained network
double weights_1[LAYER1][DIMENSION] = {
    {-0.559419, 1.204088 },
    {5.681054, 1.451364 },
    {0.346751, 0.976418 },
    {0.000000, -0.000000 },
    {-1.988339, 1.919221},
    {-0.446481, 1.457829},
    {-0.000000, -0.000000},
    {-0.134263, 0.959700},
    {0.133699, 1.119161},
    {2.010274, 1.570752},
    {-0.000000, -0.000000},
    {7.073405, -3.341986 },
    {0.000000, 0.000000},
    {2.035284, 1.318326},
    {0.420291, -2.060597},
    {-0.000000, 0.000000}};

double weights_2[LAYER2][LAYER1] = {
    {-0.254884, -0.214020, -2.747057, -0.000000, 1.619738, -0.424856, 0.000000, -0.491764, -0.330332, -3.232579, -0.000000, -2.163309, 0.000000, -3.213990, 2.866256, -0.000000},
    {0.000000, 0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.000000},
    {-0.000000, -0.000000, 0.000000, -0.000000, 0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000},
    {-0.034797, 7.473437, -0.016994, -0.000000, -1.781890, 0.250048, -0.000000, 0.513348, 0.599470, -0.059541, -0.000000, -4.548321, -0.000000, 0.474999, 1.673950, -0.000000},
    {-0.472585, 7.109786, 0.260639, 0.000000, -1.788905, -0.127643, 0.000000, 0.428724, 0.707202, 0.979315, -0.000000, -4.048883, -0.000000, 0.591226, -0.052875, -0.000000},
    {0.810772, 6.055452, 1.175967, 0.000000, 2.521729, 1.125099, 0.000000, 1.303510, 0.997026, 0.381578, -0.000000, -7.021684, 0.000000, 0.426979, -2.756504, -0.000000},
    {-0.101561, 0.106349, 0.217796, 0.000000, -0.177704, -0.192286, 0.000000, -0.434281, 0.109036, -0.052647, 0.000000, -0.482127, -0.000000, 0.362862, -0.495416, -0.000000},
    {-0.000000, 0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, -0.000000, 0.000000, -0.000000, -0.000000}
};

double weights_output_layer[OUT_LAYER] = {-4.32607544e+000, 2.59915637e-150, 1.44939016e-149, 4.93008226e+000,
                                          4.60747737e+000, 5.88102834e+000, -3.23551018e-001, 9.76717628e-129};


// Define the biases with values from pre-trained network
double bias_1[LAYER1] = {0.40546025, -4.42774979, 0.08486776, -0.2130242,
                         0.53810184, 0.56266578, -0.55623279,  0.540329,
                         0.94469664, -0.05598933, -0.25352144, -0.28231778,
                         -0.45815514, -0.13210869, 2.71258716, -0.23831385};

double bias_2[LAYER2] = {1.19471381, -0.23508044, -0.43366517,  1.19420209,
                         0.74977472, -0.22586172,  0.15849943, -0.43346352};

double bias_3 = 1.28557373;

// create arrays for storing the values of outputs from each layer
double out_layer_1[LAYER1];
double out_layer_2[LAYER2];

/*
Define the ReLu activation function that returns max(0, x).
*/
void relu(double out_layer[], int n){

    // out_layer[0] = 1.0; // bias
    int i;
    for(i=0; i<n; i++)
        out_layer[i] = out_layer[i] > 0.0 ? out_layer[i] : 0.0;
}

/*
Define the forward pass function that takes temperature and humidity as the input
for neural network and returns the result from the output layer
*/
double forward_propagation(double humidity, double temperature){

    // create an array for passing the input to the neural network (add 1.0 to the input for the bias term)
    double inputs[] = {humidity, temperature};
    double result = 0.0;

    // Run the forward pass over the neural network using the weights (includes bias in the data as a column of 1s)
    
    // perform multiplication of weights and inputs matrix
    int i, j;
    for(i=0; i<LAYER1; i++){
        out_layer_1[i] = 0.0;
        for(j=0; j<DIMENSION; j++)
            out_layer_1[i] += weights_1[i][j] * inputs[j];
    }

    // Add bias values to the outputs
    for(i=0; i<LAYER1; i++)
        out_layer_1[i] += bias_1[i];
    
    // Apply relu activation function over the outputs
    relu(out_layer_1, LAYER1);
    
    // perform multiplication of weights and inputs matrix
    for(i=0; i<LAYER2; i++){
        out_layer_2[i] = 0.0;
        for(j=0; j<LAYER1; j++)
            out_layer_2[i] += weights_2[i][j] * out_layer_1[j];
    }

    // Add bias values to the outputs
    for(i=0; i<LAYER2; i++)
        out_layer_2[i] += bias_2[i];

    // Apply relu activation function over the outputs
    relu(out_layer_2, LAYER2);

    // perform multiplication of weights and inputs matrix
    for(i=0; i<OUT_LAYER; i++){
        result += out_layer_2[i] * weights_output_layer[i];
    }

    // Add bias
    result += bias_3;

    // Apply relu over the result
    result = result > 0.0 ? result : 0.0;

    return result;
}

void setup()
{
    // defined if we dont have any other peripherial device to display the readings
    // 9600 is the baud(bps) speed with which data flows to the monitor 
    Serial.begin(9600);

    // Setup LDR sensor
    pinMode(LDR_PIN, INPUT);

    // Initialize object for LCD
    lcd.init();
    lcd.backlight();

    // Initialize communication for DHT sensors
    dht_1.begin();
    dht_2.begin();
    dht_3.begin();
    dht_4.begin();

    // Setup servo objects with the corresponding pins
    servo_1.attach(3);
    servo_2.attach(5);
    servo_3.attach(7);
    servo_4.attach(9);

    // Servo reset to zero before starting
    servo_1.write(0);
    servo_2.write(0);
    servo_3.write(0);
    servo_4.write(0);

}


void loop()
{
    delay(100);

    lcd.begin(16, 2);

    // Calculate the value of lux from LDR sensor
    int analogValue = analogRead(A0);
    float voltage = analogValue / 1024. * 5;
    float resistance = 2000 * voltage / (1 - voltage / 5);
    float lux = pow(RL10 * 1e3 * pow(10, GAMMA) / resistance, (1 / GAMMA));

    // Run the servo motors if the LDR sensors detects light (during the day)
    if(lux > 50){

        // Read the temperature and humidity data from the DHT sensors
        hum_1 = dht_1.readHumidity();
        hum_2 = dht_2.readHumidity();
        hum_3 = dht_3.readHumidity();
        hum_4 = dht_4.readHumidity();
        temp_1 = dht_1.readTemperature();
        temp_2 = dht_2.readTemperature();
        temp_3 = dht_3.readTemperature();
        temp_4 = dht_4.readTemperature();

        // Calculate the water percentage based on the temperature and humidity values
        // Temperature and humidity values are scaled using min-max normalization
        wp_1 = forward_propagation((hum_1-min_hum)/(max_hum - min_hum), (temp_1-min_temp)/(max_temp - min_temp));
        wp_2 = forward_propagation((hum_2-min_hum)/(max_hum - min_hum), (temp_2-min_temp)/(max_temp - min_temp));
        wp_3 = forward_propagation((hum_3-min_hum)/(max_hum - min_hum), (temp_3-min_temp)/(max_temp - min_temp));
        wp_4 = forward_propagation((hum_4-min_hum)/(max_hum - min_hum), (temp_4-min_temp)/(max_temp - min_temp));

        // Set the lcd print cursor and print water percentages
        lcd.setCursor(0, 0);
        lcd.print("1. "); lcd.print(wp_1); lcd.print("%");
        lcd.setCursor(9, 0);
        lcd.print(" 2. "); lcd.print(wp_2); lcd.print("%");
        lcd.setCursor(0, 1);
        lcd.print("3. "); lcd.print(wp_3); lcd.print("%");
        lcd.setCursor(9, 1);
        lcd.print(" 4. "); lcd.print(wp_4); lcd.print("%");

        // Find the angle to sweep (0-180) based on the water percentage (0-100)
        angle_1 = (wp_1 * 180) / 100;
        angle_2 = (wp_2 * 180) / 100;
        angle_3 = (wp_3 * 180) / 100;
        angle_4 = (wp_4 * 180) / 100;


        // set the angle on servos
        servo_1.write(angle_1);
        delay(5);

        servo_2.write(angle_2);
        delay(5);
        
        servo_3.write(angle_3);
        delay(5);
        
        servo_4.write(angle_4);

    }

    // If the LDR sensor detects dark then don't run the servos
    else{

        // Reset lcd cursor
        lcd.setCursor(0, 0);
        lcd.print("Dark");
        lcd.setCursor(0, 1);
        lcd.print("Irrigation Stop");

        // Set all servo angles to zero at night time
        servo_1.write(0);
        servo_2.write(0);
        servo_3.write(0);
        servo_4.write(0); 
        
    }

    delay(100);
}