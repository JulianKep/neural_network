void chain_rule(){

    double input = 2;

    //apply function a = 3x+2 -> a' = 3
    double t1 = 3 * input + 2;

    //apply function s = 1 / (1 + exp(-x)) -> s' = s * (1 - s)
    double t2 = 1 / (1 + exp(-t1));

    //apply function b = x^2 -> b' = 2x
    double t3 = pow(t2, 2.0);

    //calculate derivative of chained function at input point, use chain rule
    double result = (2 * t2) * ( (1/(1+exp(-t1))) * ( 1 - (1/(1+exp(-t1))) )) * 3;

    printf("%f\n", result);


}

void one_hidden_layer(){

    double learning_rate = 0.1;

    double target = 1;
    double input = 2.1;
    double bias = -3.8;
    double weight = 2.0;

    //calculate neuron activation -> f' = 
    double a = (input * weight) + bias;

    //use sigmoid activation
    double a_f = sigmoid(a);

    //use cost function
    double cost =  pow((a_f - target), 2.0);

    printf("Cost: %f\n", cost);


    
    //ADJUST WEIGHT (BACKPROPAGATION

    double dC_dw = (2 * (a_f - target)) * sigmoid_div(a) * input;

    printf("dC/dw%f\n", dC_dw);

    //adjust weight
    weight += learning_rate * -dC_dw;

    double a1 = (input * weight) + bias;

    double a_f1 = 1 / (1 + exp(-a1));

    double cost1 =  pow((a_f1 - target), 2.0);

    printf("New Cost: %f\n", cost1);

}