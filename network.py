
import tensorflow_addons as tfa
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Res2Net(object):
    def __init__(self, reg_rate, expansion=4, scale=4, baseWidth=26, activate=tf.nn.elu,
                #  norm=tf.keras.layers.instance_norm):
                 norm=tfa.layers.InstanceNormalization()):
        self.reg_rate = reg_rate
        self.expansion = expansion
        self.scale = scale
        self.baseWidth = baseWidth
        self.activate = activate
        self.norm = norm

    def forward(self, input_x, layer=50):
        if layer == 50:
            layers = [3, 4, 6, 3]
        if layer == 101:
            layers = [3, 4, 23, 3]
        if layer == 60:
            layers = [5, 5, 5, 5]
        input_x = self.Conv2D(input_x, 1, filters=64)
        # print('input_x:',input_x.shape)

        out = self.block(input_x, layers[0])
        # print('out:',out.shape)
        out = self.block(out, layers[1], 128)
        # print('out:',out.shape)
        out = self.block(out, layers[2], 128)
        # print('out:',out.shape)
        out = self.block(out, layers[3], 128)
        # print('out:',out.shape)

        return out

    def forward_tbm(self, input_x, temp_feats, ncol, layer=50):
        if layer == 50:
            layers = [3, 4, 6, 3]
        if layer == 101:
            layers = [3, 4, 23, 3]
        input_x = self.Conv2D(input_x, 1)

        out = self.block(input_x, layers[0])
        out = self.block(out, layers[1], 128)

        """ axial attn """
        scale = tf.sqrt(tf.cast(ncol * 64, tf.float32))
        key = self.Conv2D(temp_feats, 1)  # (N,l,l,64)
        value = self.Conv2D(temp_feats, 1)  # (N,l,l,64)
        query = self.Conv2D(out, 1)  # (1,l,l,64)
        col_attn = tf.einsum('bijd, nljd->nil', query, key) / scale
        row_attn = tf.einsum('bijd, nikd->njk', query, key) / scale
        attn = col_attn + row_attn  # (N,L,L)
        temp_weights = tf.nn.softmax(attn, axis=0, name='temp_weights')
        temp_feats_ = tf.einsum('nij,nijd->ijd', temp_weights, value)[None, ...]
        out = tf.concat([out, temp_feats_], axis=-1)

        out = self.block(out, 5, 64)
        out = self.block(out, 5, 128)

        out = self.block(out, 5, 128)
        out = self.block(out, 5, 128)

        return out

    def block(self, input_x, layers, out_channels=64):
        out = self.subblock(input_x, first=True, channels=out_channels, dilations=1)
        # print('out:',out.shape)
        d = 1
        for i in range(1, layers):
            d = 2 * d
            out = self.subblock(out, channels=out_channels, dilations=d)
            # print('out:',out.shape)

        return out

    def subblock(self, input_x, first=False, channels=64, dilations=1):
        expansion = self.expansion
        baseWidth = self.baseWidth
        scale = self.scale
        width = int(baseWidth * channels / 64)
        x = self.Conv2D(input_x, kernel_size=1, filters=width * scale)
        # print('x:',x.shape)
        outs = []
        frac = tf.cast(tf.math.ceil(tf.cast(x.shape[-1], tf.float32) / scale), tf.int32)
        for i in range(scale):
            if i == 0:
                outs.append(x[..., :frac])
            elif i == 1:
                outs.append(self.Conv2D(x[..., frac:2 * frac], kernel_size=3, filters=width, dilation=dilations))
            elif first:
                outs.append(
                    self.Conv2D(x[..., i * frac:(i + 1) * frac], kernel_size=3, filters=width, dilation=dilations))
            else:
                xi = x[..., i * frac:(i + 1) * frac] + outs[-1]
                outs.append(self.Conv2D(xi, kernel_size=3, filters=width, dilation=dilations))
        out = tf.concat(outs, axis=-1)
        out = self.Conv2D(out, kernel_size=1, filters=channels * expansion)
        if first:
            input_x = self.Conv2D(input_x, filters=expansion * channels, kernel_size=1)
            # print('input_x:',input_x.shape)

        out += input_x
        return out

    def Conv2D(self, x, kernel_size, dilation=1, filters=64, padding='same', normalize=True, activation=True):

        if normalize:
            self.norm = tfa.layers.InstanceNormalization()
            # print('x:',x.shape)
            x = self.norm(x)
        if activation:
            x = self.activate(x)

        x = tf.compat.v1.layers.conv2d(x, kernel_size=kernel_size, filters=filters, dilation_rate=dilation, padding=padding,
                             kernel_regularizer=self.reg_rate)

        return x
