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
    double bias_1_1;

} model_t;



//return map:
//[Cost, activation_after_sigmoid, activation_after_weight_and_bias]
void forward_pass(model_t *in, double* out){
    //calculate neuron activation
    double a = (in->input * in->weight1) + in->bias_1_1;

    //use sigmoid activation
    double a_f = sigmoid(a);

    //use cost function
    double cost =  pow((a_f - in->target), 2.0);

    out[0] = cost;
    out[1] = a_f;
    out[2] = a;
}


void three_hidden_layers(){

    double learning_rate = 0.01;

    model_t m = {
        .input = 2.1,
        .target = 1,
        .weight1 = 2.0,
        .bias_1_1 = -3.8,
    };


    double res[3];

    for(int iteration = 0; iteration < 1000; iteration++){

        forward_pass(&m, res);
        printf("cost: %f\n", *res);

        //ADJUST WEIGHT (BACKPROPAGATION)

        //return map:
        //[Cost, activation_after_sigmoid, activation_after_weight_and_bias]

        //(derivative of cost function w.r.t weight) * (derivative of activation function w.r.t. weight) * ...
        double dC_dw = (2 * (res[1] - m.target)) * sigmoid_div(res[2]) * m.input;

        //adjust weight
        m.weight1 += learning_rate * -dC_dw;

        printf("new weight: %f\n", m.weight1);
    }

}

//approximate sin function using one hiddne layer with 10 neurons

int main(){
    
    three_hidden_layers();

    return 0;

}
