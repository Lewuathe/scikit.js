import { DummyRegressor } from './dummyRegressor'

describe('DummyRegressor', function () {
  it('Use DummyRegressor on simple example (mean)', function () {
    const reg = new DummyRegressor()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 20, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1]
    ]

    reg.fit(X, y)
    expect(reg.predict(predictX).arraySync()).toEqual([20, 20, 20])
  })
  it('Use DummyRegressor on simple example (median)', function () {
    const reg = new DummyRegressor({ strategy: 'median' })

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 12, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1]
    ]

    reg.fit(X, y)
    expect(reg.predict(predictX).arraySync()).toEqual([12, 12, 12])
  })
  it('Use DummyRegressor on simple example (constant)', function () {
    const reg = new DummyRegressor({ strategy: 'constant', constant: 10 })

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 12, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1]
    ]

    reg.fit(X, y)
    expect(reg.predict(predictX).arraySync()).toEqual([10, 10, 10])
  })
})
