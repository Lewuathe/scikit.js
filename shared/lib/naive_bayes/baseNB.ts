import { polyfillUnique } from '../tfUtils'
import { tf } from '../../globals'
import { Scikit1D, Scikit2D } from '../types'
import { convertToNumericTensor1D, convertToNumericTensor2D, convertToTensor1D } from '../utils'
import { numericDependencies } from 'mathjs'
import { Tensor1D } from '@tensorflow/tfjs-core'

export interface NaiveBayesParams {
  priors?: Scikit1D

  varSmoothing?: number
}

export class BaseNB implements NaiveBayesParams {
  priors: NaiveBayesParams['priors']
  varSmoothing: NaiveBayesParams['varSmoothing']

  classes: tf.Tensor1D
  means: tf.Tensor1D[]
  variances: tf.Tensor1D[]

  constructor(params: NaiveBayesParams) {
    Object.assign(this, params)
    this.classes = tf.tensor1d([])
    this.means = []
    this.variances = []
  }

  public async fit(X: Scikit2D, y: Scikit1D) {
    const features = convertToNumericTensor2D(X)
    const labels = convertToTensor1D(y)

    const {values, means, variances} = tf.tidy(() => {
      console.log("tf.tidy")
      const means: Tensor1D[] = []
      const variances: Tensor1D[] = []
      polyfillUnique(tf)
      const {values, indices} = tf.unique(labels)

      tf.unstack(values).forEach((c: tf.Tensor) => {
        const mask = tf.equal(labels, c).toFloat()
        const numInstances = tf.sum(mask)
        const mean = tf.mul(features, mask.expandDims(1))
          .sum(0)
          .div(numInstances)
        const variance = tf.sub(features, mean)
          .mul(mask.expandDims(1))
          .pow(2)
          .sum(0)
          .div(numInstances)

        mean.print()
        variance.print()
        means.push(mean as Tensor1D)
        variances.push(variance as Tensor1D)
      })

      return {values, means, variances}
    })

    this.classes = values
    this.means = means
    this.variances = variances

    return this
  }

  public predictProba(X: Scikit2D) {
    const features = convertToNumericTensor2D(X)

    const probabilities = tf.tidy(() => {
      let probs: tf.Tensor1D[] = []
      this.classes.unstack().forEach((c, idx) => {
        const mean = this.means[idx]
        const variance = this.variances[idx]

        const prob = tf.sub(features, mean.expandDims(0))
            .pow(2)
            .div(variance.expandDims(0).mul(2))
            .exp()
            .div(variance.mul(2 * Math.PI).expandDims(0))
            .prod(1)
        probs.push(prob as tf.Tensor1D)
      })

      return tf.stack(probs, 1) as tf.Tensor2D
    })

    probabilities.print()
    return probabilities
  }
}