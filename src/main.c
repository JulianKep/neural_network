//single input neuron, single input neuron, configurable number of hidden layers, confirgurabel number of neurons per layer

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "../data/sinus.h"


#define HIDDEN_LAYERS 3
#define HIDDDEN_NEURONS 4


typedef struct model_t {

    double weights_i_to_h[HIDDDEN_NEURONS];

    //biases[layer_index][neuron_index]
    double biases[HIDDEN_LAYERS][HIDDDEN_NEURONS];

    //weights[from_layer_index][from_neuron_index][to_neuron_index]
    double weights[HIDDEN_LAYERS - 1][HIDDDEN_NEURONS][HIDDDEN_NEURONS];

    double weights_h_to_o[HIDDDEN_NEURONS];

    double bias_output;

} model_t;


typedef struct f_pass_return_t {

    //pre sigmoid activation in each neuron (sum of weighted inputs + bias)
    //a[layer_index][neuron_index]
    double a_p[HIDDEN_LAYERS][HIDDDEN_NEURONS];

    //post sigmoid activations
    double a_s[HIDDEN_LAYERS][HIDDDEN_NEURONS];

    double a_o;
    double cost;


} f_pass_return_t;



double sigmoid(double in){
    return (1/(1 +exp(-in)));
}

double sigmoid_div(double in){
    return (1/(1+exp(-in))) * (1-(1/(1+exp(-in))));
}

double rand_interval(int min, int max){
    int res = rand() % (max + 1 - min) + min;
    return (double)res;
}

//hidden -> number of hidden layers
//neurons -> number of neurons in each hidden layer
/* model_t construct_model(size_t hidden, size_t neurons){

    model_t result {

    }

    return result;

} */


void forward_pass(double input, double target, model_t *in, f_pass_return_t* out){

    //calculate activations of first hidden layer 
    for(size_t neuron = 0; neuron < HIDDDEN_NEURONS; neuron++){
        out->a_p[0][neuron] = (input * in->weights_i_to_h[neuron]) + in->biases[0][neuron];
    }

    for(size_t neuron = 0; neuron < HIDDDEN_NEURONS; neuron++){
        out->a_s[0][neuron] = sigmoid(out->a_p[0][neuron]);
    }

    //calculate activations other hidden layers
    for(size_t layer = 1; layer < HIDDEN_LAYERS; layer ++){

        //calcuale activation for each neuron
        for(size_t neuron = 0; neuron < HIDDDEN_NEURONS; neuron++){

            //sum over all incoming activations
            for(size_t activation = 0; activation < HIDDDEN_NEURONS; activation++){
                out->a_p[layer][neuron]+= in->weights[layer][activation][neuron] * out->a_s[layer-1][activation]; 
            }

            //add bias
            out->a_p[layer][neuron] += in->biases[layer][neuron];

            out->a_s[layer][neuron] = sigmoid(out->a_p[layer][neuron]);

        } 
    }

    //calculate final layer activation
    for(size_t activation = 0; activation < HIDDDEN_NEURONS; activation++){

        out->a_o += out->a_s[HIDDEN_LAYERS-1][activation] * in->weights_h_to_o[activation];

    }

    //add final bias
    out->a_o += in->bias_output;

    //use cost function
    out->cost = pow((out->a_o - target), 2.0);

}






double one_hidden_layer(double input, double target, double learning_rate, model_t *m){


    f_pass_return_t res;
    forward_pass(input, target, m, &res);


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


model_t m = {

    .weights_i_to_h = {1.1, -3.0, 2.0, 4.0},
    .biases = {
        {1.0, 3.0, -2.0, 4.4}, //first hidden layer
        {1.0, 1.0, -2.0, 1.4},
        {8.0, 4.0, -9.0, 4.3},
        {3.0, 7.0, -6.0, 4.4},
    },
    .weights = {
        { // from first hidden layer to second
            {2.0, -1.0, 4.2, -1.1}, // from first neuron to all other neurons in second layer
            {2.0, -1.0, 4.2, -1.1},
            {2.0, -1.0, 4.2, -1.1},
            {2.0, -1.0, 4.2, -1.1},

        },
        {
            {2.0, -1.0, 4.2, -1.1},
            {2.0, -1.0, 4.2, -1.1},
            {2.0, -1.0, 4.2, -1.1},
            {2.0, -1.0, 4.2, -1.1},

        },

    },
    .bias_output = 2.0,
    .weights_i_to_h = {1.1, -3.0, 6.0, -4.0},
};



    //loop through training data, perform stochastic gradient descent
    for(size_t epoch = 0; epoch < 1000; epoch ++){
        
        double cost;

        for(int i = 0; i < sizeof(training) / sizeof(training[0]); i++){

                cost = one_hidden_layer(training[i].x_value, training[i].y_value, 0.01, &m);
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
