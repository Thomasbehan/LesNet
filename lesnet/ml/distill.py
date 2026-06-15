"""Knowledge distillation: a small student matches a trained teacher's soft predictions.

Response-based KD (Hinton et al.): the student minimises a blend of the normal hard-label
focal loss and a temperature-softened KL term against the teacher's triage logits. This is
how the int8 live-demo model (M4.5s) keeps near-teacher accuracy at a fraction of the size.
"""
import tensorflow as tf


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, hard_loss, alpha=0.5, temperature=4.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.hard_loss = hard_loss
        self.alpha = alpha
        self.temperature = temperature
        self.teacher.trainable = False

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _distillation_loss(self, teacher_logits, student_logits):
        temperature = self.temperature
        soft_teacher = tf.nn.softmax(teacher_logits / temperature, axis=-1)
        soft_student = tf.nn.log_softmax(student_logits / temperature, axis=-1)
        per_example = -tf.reduce_sum(soft_teacher * soft_student, axis=-1)
        return tf.reduce_mean(per_example) * (temperature ** 2)

    def train_step(self, data):
        inputs, targets = data
        teacher_predictions = self.teacher(inputs, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(inputs, training=True)
            hard = self.hard_loss(targets['triage'], student_predictions['triage'])
            soft = self._distillation_loss(teacher_predictions['triage'], student_predictions['triage'])
            loss = (1.0 - self.alpha) * hard + self.alpha * soft
        trainable = self.student.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, trainable), trainable))
        self.compiled_metrics.update_state(targets['triage'], student_predictions['triage'])
        return {'loss': loss, 'hard_loss': hard, 'soft_loss': soft,
                **{metric.name: metric.result() for metric in self.metrics}}

    def test_step(self, data):
        inputs, targets = data
        student_predictions = self.student(inputs, training=False)
        loss = self.hard_loss(targets['triage'], student_predictions['triage'])
        self.compiled_metrics.update_state(targets['triage'], student_predictions['triage'])
        return {'loss': loss, **{metric.name: metric.result() for metric in self.metrics}}
