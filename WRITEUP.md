Selective time-dependant models change their B,C and delta values according to the inputs through weights that are learnt by the model
While time independant models have constant B,C and delta that are fized once the model undergoes training

The advantage of having variable parameter is that the model can differentiate between noise(whats important and not important)and update its state accordingly
The step size parameter Delta_t acts as a gate controlling how much the state updates versus how much it retains past context.When Delta_t nears infinity A_bar_t nears 0, B_bar_t nears I , the state instantly overwrites its history with current input x_t.When delta_t nears 0 a_bar_t nears I and B_bar_t nears 0, the state ignores x_t and perfectly preserves its existing hidden state h_(t-1).

This is advantageous as it can ignore irrelevant information (like padding words like the,and,but which do not provide any information) and give more importance to nouns

non-selective SSMs cant do that as A and B remain fixed, they give equal importance to all inputs which leads to them factoring noise and not able to find patterns between important information. that is, all values are treated identically regardless of weather it is a critical value token or irrelevant filler, hence non-relevant context attenuates or pollutes the state vector over long distances, making exact retrieval of arbitrary past tokens impossible.