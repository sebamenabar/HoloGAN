# lossess
def discriminator_loss(real, generated,):
    real_loss = bce(tf.ones_like(real), real)

    generated_loss = bce(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated):
    return bce(tf.ones_like(generated), generated)