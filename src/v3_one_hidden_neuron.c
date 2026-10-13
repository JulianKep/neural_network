#include <stdio.h>
#include <stdint.h>
#include <math.h>



double sigmoid(double in){
    return (1/(1 +exp(-in)));
}

double sigmoid_div(double in){
    return (1/(1+exp(-in))) * (1-(1/(1+exp(-in))));
}

typedef struct model_t
{
    double input;
    double target;

    double weight1;
    double weight2;
    double bias_1_1;
    double bias_1_2;

} model_t;



void forward_pass(model_t *in, double* out){

    //hidden neuron activation
    double a_1 = (in->input * in->weight1) + in->bias_1_1;

    //sigmoid activation
    double a_2 = sigmoid(a_1);

    //output neuron activation
    double a_3 = (a_2 * in->weight2) + in->bias_1_2;

    //use cost function
    double cost =  pow((a_3 - in->target), 2.0);

    out[0] = cost;
    out[1] = a_1;
    out[2] = a_2;
    out[3] = a_3;
}


void three_hidden_layers(double input, double target){

    double learning_rate = 0.01;

    model_t m = {
        .input = input,
        .target = target,

        .weight1 = 2.0,
        .bias_1_1 = -3.8,
        .weight2 = 1.0,
        .bias_1_2 = -1.6,
    };


    double res[4];

    for(int iteration = 0; iteration < 100; iteration++){


        forward_pass(&m, res);
        printf("cost: %f\n", *res);

        //ADJUST WEIGHT (BACKPROPAGATION)

        //return map:
        //[Cost, activation_after_sigmoid, activation_after_weight_and_bias]

        //adjust w1
        double dC_dw1 = (2 * (res[3] - m.target)) * m.weight2 * sigmoid_div(res[1]) * m.input;

        m.weight1 += learning_rate * -dC_dw1;
    
        //adjust w2
        double dC_dw2 = (2 * (res[3] - m.target)) * res[2];

        m.weight2 += learning_rate * -dC_dw2;

        //adjust b1
        double dC_db1 = (2 * (res[3] - m.target)) * m.weight2 * sigmoid_div(res[1]);

        m.bias_1_1 += learning_rate * -dC_db1;

        //adjust b2
        double dC_db2 = (2 * (res[3] - m.target));

        m.bias_1_1 += learning_rate * -dC_db2;



    }

}



//approximate sin function using one hiddne layer with 10 neurons

int main(){
    
    three_hidden_layers(4, 1);

    return 0;

}
