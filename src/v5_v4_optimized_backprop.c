#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "../data/sinus.h"



//weight member name carries information of what neuron it is attached to
//double weight_i_to_1_2; -> weight from input neuron to neron in hidden layer , second neuron from the top
typedef struct model_t
{
    double input;
    double target;

    double weight_i_to_1_1;
    double bias_1_1;
    double weight_i_to_1_2;
    double bias_1_2;
    double weight_i_to_1_3;
    double bias_1_3;

    double weight_1_1_to_o;
    double weight_1_2_to_o;
    double weight_1_3_to_o;

    double bias_output;

} model_t;


double sigmoid(double in){
    return (1/(1 +exp(-in)));
}

double sigmoid_div(double in){
    return (1/(1+exp(-in))) * (1-(1/(1+exp(-in))));
}


void forward_pass(double input, double target, model_t *in, double* out){

    //hidden neuron activations. a_layer_neuron
    double a_1_1 = (input * in->weight_i_to_1_1) + in->bias_1_1;
    double a_1_2 = (input * in->weight_i_to_1_2) + in->bias_1_2;
    double a_1_3 = (input * in->weight_i_to_1_3) + in->bias_1_3;

    //sigmoid activations
    double sa_1 = sigmoid(a_1_1);
    double sa_2 = sigmoid(a_1_2);
    double sa_3 = sigmoid(a_1_3);

    //output neuron activation
    double a_o = (sa_1 * in->weight_1_1_to_o) + (sa_2 * in->weight_1_2_to_o) + (sa_3 * in->weight_1_3_to_o) + in->bias_output;

    //use cost function
    double cost =  pow((a_o - target), 2.0);

    out[0] = cost;
    out[1] = a_1_1;
    out[2] = a_1_2;
    out[3] = a_1_3;
    out[4] = sa_1;
    out[5] = sa_2;
    out[6] = sa_3;
    out[7] = a_o;
}


double three_hidden_layers(double input, double target, double learning_rate, model_t *m){



    double res[8];
    forward_pass(input, target, m, res);

    //printf("cost: %f\n", *res);

    ////CALCULATE GRADIENTS

    //output bias
    double dC_db_o = (2 * (res[7] - target));

    //hidden layer weights to output
    double dC_dw_1_1_to_o = dC_db_o * res[4];
    double dC_dw_1_2_to_o = dC_db_o * res[5];
    double dC_dw_1_3_to_o = dC_db_o * res[6];



    //hidden layer biases
    double dC_db_1_1 = dC_db_o * m->weight_1_1_to_o * sigmoid_div(res[1]);
    double dC_db_1_2 = dC_db_o * m->weight_1_2_to_o * sigmoid_div(res[2]);
    double dC_db_1_3 = dC_db_o * m->weight_1_3_to_o * sigmoid_div(res[3]);

    //input to hidden layer weights
    double dC_dw_i_to_1_1 = dC_db_1_1 * input;
    double dC_dw_i_to_1_2 = dC_db_1_2 * input;
    double dC_dw_i_to_1_3 = dC_db_1_3 * input;


    ////PERFORM GRADIENT DESCENT
    m->bias_output += learning_rate * -dC_db_o;

    m->weight_1_1_to_o += learning_rate * -dC_dw_1_1_to_o;
    m->weight_1_2_to_o += learning_rate * -dC_dw_1_2_to_o;
    m->weight_1_3_to_o += learning_rate * -dC_dw_1_3_to_o;

    m->bias_1_1 += learning_rate * -dC_db_1_1;
    m->bias_1_2 += learning_rate * -dC_db_1_2;
    m->bias_1_3 += learning_rate * -dC_db_1_3;

    m->weight_i_to_1_1 += learning_rate * -dC_dw_i_to_1_1;
    m->weight_i_to_1_2 += learning_rate * -dC_dw_i_to_1_2;
    m->weight_i_to_1_3 += learning_rate * -dC_dw_i_to_1_3;

    return res[0];


}


int main(){


model_t m = {
    .weight_i_to_1_1 = 2.0,
    .bias_1_1 = -3.8,
    .weight_i_to_1_2 = 1.0,
    .bias_1_2 = -1.6,
    .weight_i_to_1_3 = 2.0,
    .bias_1_3 = -9.0,

    .weight_1_1_to_o = 2.4,
    .weight_1_2_to_o = 2.4,
    .weight_1_3_to_o = 2.4,

    .bias_output = 1.0,
};

    //loop through training data, perform stochastic gradient descent
    for(size_t epoch = 0; epoch < 100000; epoch ++){
        
        double cost;

        for(int i = 0; i < sizeof(training) / sizeof(training[0]); i++){

                cost = three_hidden_layers(training[i].x_value, training[i].y_value, 0.01, &m);
        }

        if(epoch % 10 == 0){
            printf("%f\n", cost);
        }
}

    double out[8];
    forward_pass(0.5, 0.479, &m, out);
    printf("final activation: %f\n", out[7]);
    printf("cost: %f\n", out[0]);
    

    return 0;

}
