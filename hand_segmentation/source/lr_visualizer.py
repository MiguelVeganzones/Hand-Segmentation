import matplotlib.pyplot as plt
import tensorflow as tf

def plot_scheduler(step, schedulers):
    if not isinstance(schedulers, list):
        schedulers = [schedulers]

    for scheduler in schedulers:
        plt.plot(range(step), scheduler(range(step)))
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
    xticks = [i*2411*5 for i in range(30//5+1)]
    xtick_labels = [str(i*5) for i in range(30//5+1)]
    plt.xticks(xticks, xtick_labels)
    plt.legend(["Constant", "Linear decay", "Exp_1", "Exp_2", "Cosine", "Square root"])
    plt.show()

if __name__ == "__main__":
    initial_learning_rate = 0.0015
    final_learning_rate = 0.0005
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/30)
    steps_per_epoch = int(2411)


    lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = 0.001,
                decay_steps = 30 * 2411,
                end_learning_rate=0.001,
                power=1.0,
                cycle=False,
                name=None
            )

    lr0 = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=steps_per_epoch, 
                    decay_rate=learning_rate_decay_factor,
                    staircase=True
                  )

    initial_learning_rate = 0.0017
    final_learning_rate = 0.0007
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/30)
    steps_per_epoch = int(2411)
    lr1 = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=steps_per_epoch, 
                decay_rate=learning_rate_decay_factor,
                staircase=True
                )

    lr2 = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate = 0.0018,
                        decay_steps = 30 * 2411,
                        alpha = 0.2
                       )

    initial_learning_rate = 0.0017
    final_learning_rate = 0.0006
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/30)
    steps_per_epoch = int(2411)
    lr3 = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate = initial_learning_rate,
                        decay_steps = 30 * 2411,
                        end_learning_rate=final_learning_rate,
                        power=1.0,
                        cycle=False,
                        name=None
                    )

    #initial_learning_rate = 0.002
    #lr4 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #    initial_learning_rate,
    #    2411 * 5,
    #    t_mul=2,
    #    m_mul=0.7,
    #    alpha=0.2,
    #    name=None
    #)

    initial_learning_rate = 0.0017
    final_learning_rate = 0.0004
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/30)
    steps_per_epoch = int(2411)
    lr5 = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate = initial_learning_rate,
                        decay_steps = 30 * 2411,
                        end_learning_rate=final_learning_rate,
                        power=.5,
                        cycle=False,
                        name=None
                    )

    plot_scheduler(30 * 2411 , [lr, lr3, lr0, lr1, lr2, lr5])


