---
title: "React ile Chart.js Kullanımı"
categories:
  - Yazılım
header:
  teaser: https://lh3.googleusercontent.com/W3zBiYKVQGDy9HoC4bxw1SZZtnEC3sSwzBvbyJV5KvtuWS2N8x2As_U-xsXxEqj9vm9Tldq5ihVdMaWko580IEe1PmFkjmSl8GNBHwnqQOPJ4wfnV2DRBQFKog71GS_g-47wqoM0
---
![React ile Chart.js Kullanımı](https://lh3.googleusercontent.com/W3zBiYKVQGDy9HoC4bxw1SZZtnEC3sSwzBvbyJV5KvtuWS2N8x2As_U-xsXxEqj9vm9Tldq5ihVdMaWko580IEe1PmFkjmSl8GNBHwnqQOPJ4wfnV2DRBQFKog71GS_g-47wqoM0)

```
npm i chart.js react-chartjs-2
```

Sonrasında, istediğimiz tipi import ediyoruz:

```
import { Bar, Line, Pie, Radar, Bubble } from "react-chartjs-2";
```

Verilerimizi de aktarıyoruz. Örnek kullanım:

```
const data = {
  labels: ["January", "February", "March", "April", "May", "June", "July"],
  datasets: [
    {
      label: "Birinci Veri Seti",
      fill: true,
      lineTension: 0.1,
      backgroundColor: "rgba(75,192,192,0.4)",
      borderColor: "rgba(75,192,192,1)",
      borderCapStyle: "butt",
      borderDash: [],
      borderDashOffset: 0.0,
      borderJoinStyle: "miter",
      pointBorderColor: "rgba(75,192,192,1)",
      pointBackgroundColor: "#fff",
      pointBorderWidth: 1,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(75,192,192,1)",
      pointHoverBorderColor: "rgba(220,220,220,1)",
      pointHoverBorderWidth: 2,
      pointRadius: 1,
      pointHitRadius: 10,
      data: [65, 59, 80, 81, 56, 55, 40]
    }
  ]
};

class Charts extends Component {
  render() {
    return (
      <div>
        <h2 className="margin-auto">
          Charts
        </h2>
        <br />
        <div className="chart-container">
          <Radar width={50} height={25} data={data} />
          <Pie width={50} height={25} ref="chart" data={data} />
          <Bar width={50} height={25} ref="chart" data={data} />
          <Bubble width={50} height={25} ref="chart" data={data} />
        </div>
      </div>
    );
  }

  componentDidMount() {
    const { datasets } = this.refs.chart.chartInstance.data;
    console.log(datasets[0].data);
  }
}

export default Charts;
```
