import { BaseNB } from './baseNB'

describe.only('BaseNaiveBayes', function () {
    it('correctly predicts sklearn example', async () => {
      const X = [[0.1, 0.9], [0.3, 0.7], [0.9, 0.1], [0.8, 0.2], [0.81, 0.19]]
      const y = [0, 0, 1, 1, 1]

      const model = new BaseNB({ priors: [0.5, 0.5] })

      await model.fit(X, y)
      const labels = model.predict(X)

      expect(labels.arraySync()).toEqual([0, 0, 1, 1, 1])
    })
  })
