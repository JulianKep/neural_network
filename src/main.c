#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "../data/sinus.h"

#define HIDDDEN 5

//weight member name carries information of what neuron it is attached to
//double weight_i_to_1_2; -> weight from input neuron to neron in hidden layer , second neuron from the top
typedef struct model_t {

    double weights_i_to_h[HIDDDEN];
    double biases_h[HIDDDEN];
    double weights_h_to_o[HIDDDEN];

    double bias_output;

} model_t;

typedef struct f_pass_return_t {

    double a_1_n[HIDDDEN];
    double sa_n[HIDDDEN];
    double a_o;
    double cost;


} f_pass_return_t;



double sigmoid(double in){
    return (1/(1 +exp(-in)));
}

double sigmoid_div(double in){
    return (1/(1+exp(-in))) * (1-(1/(1+exp(-in))));
}


void forward_pass(double input, double target, model_t *in, f_pass_return_t* out){

    //hidden neuron activations. a_layer_neuron
    for(int i = 0; i < HIDDDEN; i++){
        out->a_1_n[i] = (input * in->weights_i_to_h[i]) + in->biases_h[i];
    }

    //sigmoid activations
    for(int i = 0; i < HIDDDEN; i++){
        out->sa_n[i] = sigmoid(out->a_1_n[i]);
    }

    //output neuron activation
    double sum_weight_activation_products;

    for(int i = 0; i < HIDDDEN; i++){

        out->sa_n[i] = sigmoid(out->a_1_n[i]);

        sum_weight_activation_products += out->sa_n[i] * in->weights_h_to_o[i];
    }

    out->a_o = sum_weight_activation_products + in->bias_output;
    

    //use cost function
    out->cost = pow((out->a_o - target), 2.0);
    
}


double one_hidden_layer(double input, double target, double learning_rate, model_t *m){



    f_pass_return_t res;


    forward_pass(input, target, m, &res);

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

    .weights_i_to_h = {2.0, 1.0, 2.1},
    .biases_h =       {1.1, 4.1, 3.0},
    .weights_h_to_o = {4.3, 2.2, 1.0},

    .bias_output =     1.0,
};

    //loop through training data, perform stochastic gradient descent
    for(size_t epoch = 0; epoch < 100000; epoch ++){
        
        double cost;

        for(int i = 0; i < sizeof(training) / sizeof(training[0]); i++){

                cost = one_hidden_layer(training[i].x_value, training[i].y_value, 0.01, &m);
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
