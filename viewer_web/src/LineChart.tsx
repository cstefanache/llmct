import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export interface LineSeries {
  label: string;
  data: number[];
  stroke: string;
}

export function LineChart(props: {
  xs: number[];
  series: LineSeries[];
  title?: string;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const uref = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const data: uPlot.AlignedData = [
      props.xs,
      ...props.series.map((s) => s.data),
    ];
    const opts: uPlot.Options = {
      title: props.title,
      width: el.clientWidth || 600,
      height: props.height ?? 320,
      series: [
        { label: "layer" },
        ...props.series.map((s) => ({ label: s.label, stroke: s.stroke, width: 1.4 })),
      ],
      legend: { live: false },
      scales: { x: { time: false } },
      axes: [
        { label: "layer" },
        {},
      ],
    };
    uref.current = new uPlot(opts, data, el);
    const onResize = () => uref.current?.setSize({ width: el.clientWidth, height: props.height ?? 320 });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      uref.current?.destroy();
      uref.current = null;
    };
  }, [props.xs, props.series, props.title, props.height]);

  return <div ref={ref} className="chart" />;
}
