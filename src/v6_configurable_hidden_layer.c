#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "../data/sinus.h"

#define HIDDDEN_NEURONS 10

//weight member name carries information of what neuron it is attached to
//double weight_i_to_1_2; -> weight from input neuron to neron in hidden layer , second neuron from the top
typedef struct model_t {

    double weights_i_to_h[HIDDDEN_NEURONS];
    double biases_h[HIDDDEN_NEURONS];
    double weights_h_to_o[HIDDDEN_NEURONS];

    double bias_output;

} model_t;

typedef struct f_pass_return_t {

    double a_1_n[HIDDDEN_NEURONS];
    double sa_n[HIDDDEN_NEURONS];
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
    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        out->a_1_n[i] = (input * in->weights_i_to_h[i]) + in->biases_h[i];
    }

    //sigmoid activations
    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        out->sa_n[i] = sigmoid(out->a_1_n[i]);
    }

    //output neuron activation
    double sum_weight_activation_products = 0;

    for(int i = 0; i < HIDDDEN_NEURONS; i++){

        out->sa_n[i] = sigmoid(out->a_1_n[i]);

        sum_weight_activation_products += out->sa_n[i] * in->weights_h_to_o[i];
    }

    out->a_o = sum_weight_activation_products + in->bias_output;
    

    //use cost function
    out->cost = pow((out->a_o - target), 2.0);
    
}


double three_hidden_layers(double input, double target, double learning_rate, model_t *m){



    f_pass_return_t res;


    forward_pass(input, target, m, &res);

    //printf("cost: %f\n", *res);

    ////CALCULATE GRADIENTS

    //output bias
    double dC_db_o = (2 * (res.a_o - target));


    //hidden layer weights to output
    double dc_dw_1_n_to_o[HIDDDEN_NEURONS];

    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        dc_dw_1_n_to_o[i] = dC_db_o * res.sa_n[i];
    }


    //hidden layer biases
    double dC_db_1_n[HIDDDEN_NEURONS];
    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        dC_db_1_n[i] = dC_db_o * m->weights_h_to_o[i] * sigmoid_div(res.a_1_n[i]);
    }

    //input to hidden layer weights
    double dC_dw_i_to_1_n[HIDDDEN_NEURONS];
    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        dC_dw_i_to_1_n[i] = dC_db_1_n[i] * input;
    }


    ////PERFORM GRADIENT DESCENT
    m->bias_output += learning_rate * -dC_db_o;

    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        m->weights_h_to_o[i] += learning_rate * -dc_dw_1_n_to_o[i];
    }

    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        m->biases_h[i] += learning_rate * -dC_db_1_n[i];
    }

    for(int i = 0; i < HIDDDEN_NEURONS; i++){
        m->weights_i_to_h[i] += learning_rate * -dC_dw_i_to_1_n[i];
    }

    return res.cost;


}


int main(){


/* model_t m = {

    .weights_i_to_h = {2.0, 1.0, 2.1, 1.0, 2.1},
    .biases_h =       {1.1, 4.1, 3.0, 1.0, 2.1},
    .weights_h_to_o = {4.3, 2.2, 1.0, 1.0, 1.1},

    .bias_output =     1.0,
};
 */
model_t m = {

    .weights_i_to_h = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5},
    .biases_h       = {-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5},
    .weights_h_to_o = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0},

    .bias_output = 1.0,
};

    //loop through training data, perform stochastic gradient descent
    for(size_t epoch = 0; epoch < 1000; epoch ++){
        
        double cost;

        for(int i = 0; i < sizeof(training) / sizeof(training[0]); i++){

                cost = three_hidden_layers(training[i].x_value, training[i].y_value, 0.01, &m);
        }

        if(epoch % 1000 == 0){
            printf("%f\n", cost);
        }
}

    f_pass_return_t out;
    forward_pass(0.5, 0.479, &m, &out);
    printf("final activation: %f\n", out.a_o);
    printf("cost: %f\n", out.cost);
    

    return 0;

}
