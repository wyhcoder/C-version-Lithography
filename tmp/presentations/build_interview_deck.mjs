import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";
import { pathToFileURL } from "node:url";

const SKILL_DIR = "/Users/wangyuhang/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const TEMPLATE = "/Users/wangyuhang/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx";
const WORKSPACE = "/Users/wangyuhang/Desktop/school/Litho_cpp";
const BUILD_DIR = path.join(WORKSPACE, "tmp/presentations/build");
const PREVIEW_DIR = path.join(BUILD_DIR, "preview");
const CANDIDATE = path.join(BUILD_DIR, "candidate.pptx");

const { makeNativeBulletParagraphs } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);

const FONT = "PingFang SC";
const C = {
  bg: "#FBFCFE",
  ink: "#102238",
  body: "#405266",
  muted: "#718096",
  blue: "#155C93",
  cyan: "#3A9BD1",
  pale: "#EAF3F9",
  pale2: "#F2F6F9",
  line: "#D6E1E8",
  white: "#FFFFFF",
  warm: "#F0A35E",
};

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(TEMPLATE));
const keepIdx = [0, 4, 7, 10, 16, 18, 25];
const kept = keepIdx.map((i) => presentation.slides.getItem(i));
for (let i = presentation.slides.count - 1; i >= 0; i -= 1) {
  if (!keepIdx.includes(i)) presentation.slides.remove(i);
}
// Put the timeline before the two technical project detail slides.
kept[4].moveTo(2);

function reset(slide) {
  slide.shapes.deleteAll();
  slide.background.fill = C.bg;
}

function addText(slide, text, position, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.body,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "none",
    wrap: "square",
    lineSpacing: options.lineSpacing ?? 1.08,
    insets: options.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return box;
}

function addTitle(slide, title, page) {
  addText(slide, title, { left: 54, top: 42, width: 1110, height: 58 }, {
    fontSize: 42, bold: true, color: C.ink, name: "slide-title",
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 54, top: 112, width: 58, height: 5 },
    fill: C.cyan,
    line: { fill: "none", width: 0 },
  });
  addText(slide, String(page).padStart(2, "0"), { left: 1170, top: 650, width: 54, height: 24 }, {
    fontSize: 15, color: C.muted, alignment: "right", name: "page-number",
  });
}

function addFooter(slide, text = "王宇航｜技术面试汇报") {
  addText(slide, text, { left: 54, top: 650, width: 500, height: 24 }, {
    fontSize: 14, color: C.muted, name: "footer",
  });
}

function addCard(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: options.fill ?? C.white,
    line: { style: "solid", fill: options.line ?? C.line, width: options.lineWidth ?? 1 },
    borderRadius: options.radius ?? 18,
    shadow: options.shadow ?? "shadow-none",
    name: options.name,
  });
}

function addBulletList(slide, items, position, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
    name: options.name,
  });
  box.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: options.marginLeftPoints ?? 17,
    hangingPoints: options.hangingPoints ?? 8,
    spaceAfterPoints: options.spaceAfterPoints ?? 8,
  });
  box.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 22,
    color: options.color ?? C.body,
    autoFit: "none",
    wrap: "square",
    lineSpacing: options.lineSpacing ?? 1.06,
    insets: { top: 0, right: 2, bottom: 0, left: 0 },
  };
  return box;
}

function addMetric(slide, x, value, label, accent = C.blue) {
  addCard(slide, { left: x, top: 452, width: 346, height: 150 }, { fill: C.pale2, line: C.pale2 });
  addText(slide, value, { left: x + 22, top: 472, width: 302, height: 54 }, {
    fontSize: 42, bold: true, color: accent,
  });
  addText(slide, label, { left: x + 22, top: 536, width: 302, height: 46 }, {
    fontSize: 19, color: C.body,
  });
}

const slide1 = presentation.slides.getItem(0);
reset(slide1);
slide1.shapes.add({
  geometry: "rect",
  position: { left: 0, top: 0, width: 18, height: 720 },
  fill: C.blue,
  line: { fill: "none", width: 0 },
});
slide1.shapes.add({
  geometry: "roundRect",
  position: { left: 760, top: 102, width: 424, height: 470 },
  fill: C.pale,
  line: { fill: "none", width: 0 },
  borderRadius: 28,
});
slide1.images.add({
  blob: await fs.readFile(path.join(WORKSPACE, "outputs/sraf_geometry/sraf_fitted_curve_overlay.png")),
  contentType: "image/png",
  alt: "SRAF骨架线参数化曲线拟合示例",
  fit: "contain",
  geometry: "roundRect",
  borderRadius: 22,
  position: { left: 790, top: 132, width: 364, height: 364 },
});
addText(slide1, "技术面试汇报", { left: 72, top: 64, width: 320, height: 36 }, {
  fontSize: 20, bold: true, color: C.blue,
});
addText(slide1, "个人技术经历\n与项目汇报", { left: 72, top: 190, width: 610, height: 178 }, {
  fontSize: 56, bold: true, color: C.ink, lineSpacing: 0.94,
});
addText(slide1, "曲线 OPC · 反演光刻 · 参数化优化", { left: 74, top: 390, width: 600, height: 38 }, {
  fontSize: 24, color: C.body,
});
addText(slide1, "王宇航\n广东工业大学｜控制工程\n应聘岗位：算法研发 / EDA研发（按JD修改）", { left: 74, top: 505, width: 610, height: 122 }, {
  fontSize: 21, color: C.body, lineSpacing: 1.18,
});
addText(slide1, "8分钟版本", { left: 1012, top: 615, width: 172, height: 30 }, {
  fontSize: 15, color: C.muted, alignment: "right",
});
slide1.speakerNotes.textFrame.setText(
  "开场建议：各位面试官好，我叫王宇航，目前就读于广东工业大学控制工程专业。我的研究主要围绕反演光刻、曲线OPC和SRAF参数化优化展开。接下来我会重点介绍两段科研工作和一段腾讯量子实验室实习经历。"
);

const slide2 = presentation.slides.getItem(1);
reset(slide2);
addTitle(slide2, "教育背景与能力结构", 2);
addFooter(slide2);
addText(slide2, "教育背景", { left: 62, top: 162, width: 430, height: 36 }, {
  fontSize: 24, bold: true, color: C.blue,
});
addText(slide2, "广东工业大学", { left: 62, top: 218, width: 470, height: 42 }, {
  fontSize: 31, bold: true, color: C.ink,
});
addText(slide2, "2024.09 - 至今｜控制工程 硕士", { left: 62, top: 270, width: 490, height: 32 }, {
  fontSize: 21, color: C.body,
});
addText(slide2, "研究方向", { left: 62, top: 352, width: 430, height: 34 }, {
  fontSize: 24, bold: true, color: C.blue,
});
addText(slide2, "曲线 OPC、SRAF 放置、反演光刻技术", { left: 62, top: 405, width: 500, height: 62 }, {
  fontSize: 25, bold: true, color: C.ink,
});
addCard(slide2, { left: 62, top: 510, width: 470, height: 92 }, { fill: C.pale, line: C.pale });
addText(slide2, "专业成绩 3.93 / 5\n2025学年研究生一等奖学金", { left: 86, top: 530, width: 420, height: 58 }, {
  fontSize: 20, color: C.body, lineSpacing: 1.16,
});
slide2.shapes.add({
  geometry: "line",
  position: { left: 624, top: 170, width: 0, height: 420 },
  fill: "none",
  line: { style: "solid", fill: C.line, width: 1.5 },
});
addText(slide2, "能力结构", { left: 682, top: 162, width: 470, height: 36 }, {
  fontSize: 24, bold: true, color: C.blue,
});
const skills = [
  ["物理建模", "傅里叶光学与光刻成像模型"],
  ["数值优化", "Level-set、WENO、EPE、MEEF"],
  ["工程实现", "Python算法验证、C++面向对象实现"],
  ["数据建模", "批量仿真、数据集建设、正向与逆向预测"],
];
for (let i = 0; i < skills.length; i += 1) {
  const y = 226 + i * 94;
  addText(slide2, skills[i][0], { left: 682, top: y, width: 156, height: 34 }, {
    fontSize: 22, bold: true, color: C.ink,
  });
  addText(slide2, skills[i][1], { left: 850, top: y + 1, width: 350, height: 58 }, {
    fontSize: 19, color: C.body,
  });
  if (i < skills.length - 1) {
    slide2.shapes.add({
      geometry: "line",
      position: { left: 682, top: y + 67, width: 506, height: 0 },
      fill: "none",
      line: { style: "solid", fill: C.line, width: 1 },
    });
  }
}
slide2.speakerNotes.textFrame.setText(
  "这一页控制在40秒。重点说明研究方向和能力结构，不逐条复述简历。可以补充：我的特点是能够把物理模型、优化算法和工程实现串联起来，并且既能用Python快速验证，也能用C++完成性能导向的实现。"
);

const slide3 = presentation.slides.getItem(2);
reset(slide3);
addTitle(slide3, "从物理建模到工程预测", 3);
addFooter(slide3);
addText(slide3, "研究与实习围绕参数化、优化和自动化展开", { left: 56, top: 135, width: 800, height: 34 }, {
  fontSize: 22, color: C.body,
});
slide3.shapes.add({
  geometry: "line",
  position: { left: 108, top: 330, width: 1000, height: 0 },
  fill: "none",
  line: { style: "solid", fill: C.line, width: 3 },
});
const milestones = [
  {
    x: 108, date: "2024.09 - 2025.05", title: "Level-set 反演光刻",
    detail: "光刻成像模型\nWENO空间离散\n模式误差驱动轮廓演化",
  },
  {
    x: 460, date: "2025.06 - 2026.03", title: "参数化曲线 OPC",
    detail: "轮廓抽稀与曲线拟合\nMEEF控制点优化\nSRAF骨架线与宽度表达",
  },
  {
    x: 812, date: "2026.04 - 2026.07", title: "腾讯量子实验室",
    detail: "参数化批量仿真\n正向与逆向预测\n设计流程自动化",
  },
];
for (const m of milestones) {
  slide3.shapes.add({
    geometry: "ellipse",
    position: { left: m.x, top: 319, width: 22, height: 22 },
    fill: C.blue,
    line: { fill: C.white, width: 3 },
  });
  addText(slide3, m.date, { left: m.x, top: 254, width: 290, height: 32 }, {
    fontSize: 18, bold: true, color: C.blue,
  });
  addText(slide3, m.title, { left: m.x, top: 374, width: 300, height: 42 }, {
    fontSize: 26, bold: true, color: C.ink,
  });
  addText(slide3, m.detail, { left: m.x, top: 430, width: 306, height: 118 }, {
    fontSize: 20, color: C.body, lineSpacing: 1.2,
  });
}
slide3.speakerNotes.textFrame.setText(
  "这一页先建立主线：第一阶段解决像素级曲线掩模生成，第二阶段把高冗余结果转成可制造、可优化的参数化表达，第三阶段将参数化和自动化经验应用到量子器件仿真与预测。"
);

const slide4 = presentation.slides.getItem(3);
reset(slide4);
addTitle(slide4, "基于水平集的反演光刻", 4);
addFooter(slide4);
addText(slide4, "项目目标", { left: 58, top: 160, width: 250, height: 34 }, {
  fontSize: 23, bold: true, color: C.blue,
});
addText(slide4, "通过 ILT 提高复杂版图的成像质量，并生成曲线掩模与 SRAF。", { left: 58, top: 205, width: 512, height: 78 }, {
  fontSize: 24, bold: true, color: C.ink, lineSpacing: 1.12,
});
addText(slide4, "个人工作", { left: 58, top: 318, width: 250, height: 34 }, {
  fontSize: 23, bold: true, color: C.blue,
});
addBulletList(slide4, [
  "基于傅里叶光学构建光刻成像模型",
  "以模式误差作为速度场驱动轮廓演化",
  "采用WENO格式完成空间离散",
  "完成Python与C++双平台实现",
], { left: 58, top: 364, width: 518, height: 210 }, { fontSize: 21, spaceAfterPoints: 9 });
const sourceImage = slide4.images.items[0];
sourceImage.replace({
  blob: await fs.readFile(path.join(WORKSPACE, "build-release/result/multi_view.png")),
  contentType: "image/png",
  alt: "项目运行结果中的目标图形、空中像和掩模视图",
  fit: "contain",
});
sourceImage.frame = { left: 620, top: 150, width: 590, height: 420 };
sourceImage.crop = { left: 0, top: 0, right: 0, bottom: 0 };
sourceImage.geometry = "roundRect";
sourceImage.borderRadius = 16;
addText(slide4, "项目运行结果示例：目标、空中像与掩模", { left: 620, top: 586, width: 590, height: 30 }, {
  fontSize: 16, color: C.muted, alignment: "center",
});
slide4.speakerNotes.textFrame.setText(
  "讲解顺序：问题背景、个人职责、算法链路、工程实现。可补充说明速度场符号如何决定轮廓向内或向外运动，以及WENO在保持界面稳定和处理尖锐边界方面的作用。图片来自本地项目运行结果。"
);

const slide5 = presentation.slides.getItem(4);
reset(slide5);
addTitle(slide5, "参数化曲线 OPC 与 SRAF", 5);
addFooter(slide5);
addText(slide5, "将高冗余的像素版图转化为“主图形控制点 + SRAF骨架线控制点 + 宽度”的轻量化表达。", { left: 56, top: 136, width: 1166, height: 56 }, {
  fontSize: 23, bold: true, color: C.ink,
});
addCard(slide5, { left: 56, top: 214, width: 405, height: 286 }, { fill: "#0A0A0A", line: C.line, radius: 16 });
slide5.images.add({
  blob: await fs.readFile(path.join(WORKSPACE, "outputs/sraf_geometry/sraf_skeleton_overlay.png")),
  contentType: "image/png",
  alt: "SRAF骨架线与控制点叠加图",
  fit: "contain",
  geometry: "roundRect",
  borderRadius: 14,
  position: { left: 67, top: 225, width: 383, height: 264 },
});
addCard(slide5, { left: 490, top: 214, width: 732, height: 286 }, { fill: C.white, line: C.line, radius: 16 });
slide5.images.add({
  blob: await fs.readFile(path.join(WORKSPACE, "outputs/sraf_optimizer_init/pv_band_comparison.png")),
  contentType: "image/png",
  alt: "SRAF宽度优化前后的PVBand对比",
  fit: "contain",
  geometry: "roundRect",
  borderRadius: 14,
  position: { left: 502, top: 225, width: 708, height: 264 },
});
addText(slide5, "骨架线与控制点", { left: 56, top: 510, width: 405, height: 30 }, {
  fontSize: 17, color: C.muted, alignment: "center",
});
addText(slide5, "SRAF宽度优化与PVBand对比", { left: 490, top: 510, width: 732, height: 30 }, {
  fontSize: 17, color: C.muted, alignment: "center",
});
const contributions = [
  ["轮廓抽稀", "等距采样、DP简化、曲率采样"],
  ["控制点优化", "EPE与控制点位移构建MEEF矩阵"],
  ["SRAF表达", "骨架线控制点与宽度分阶段优化"],
];
for (let i = 0; i < contributions.length; i += 1) {
  const x = 56 + i * 389;
  addText(slide5, contributions[i][0], { left: x, top: 558, width: 150, height: 30 }, {
    fontSize: 20, bold: true, color: C.blue,
  });
  addText(slide5, contributions[i][1], { left: x, top: 591, width: 350, height: 46 }, {
    fontSize: 17, color: C.body,
  });
}
slide5.speakerNotes.textFrame.setText(
  "这一页是科研重点，建议讲两分钟。先说明像素版图的数据冗余和制造表达问题，再讲轮廓抽稀、控制点优化和SRAF轻量化表达。图中展示了骨架线控制点以及宽度优化前后的PVBand结果。没有在简历中写明的具体EPE下降比例不要临时补数字。"
);

const slide6 = presentation.slides.getItem(5);
reset(slide6);
addTitle(slide6, "腾讯量子实验室：仿真数据与模型训练", 6);
addFooter(slide6);
addText(slide6, "搭建几何参数生成、批量仿真、数据收集和模型训练的自动化流程。", { left: 56, top: 135, width: 1120, height: 42 }, {
  fontSize: 23, bold: true, color: C.ink,
});
const flowLabels = ["几何参数生成", "批量仿真", "数据集建设", "模型训练", "正向 / 逆向预测"];
const flowXs = [56, 292, 528, 764, 1000];
for (let i = 0; i < flowLabels.length; i += 1) {
  addCard(slide6, { left: flowXs[i], top: 224, width: 178, height: 92 }, {
    fill: i === flowLabels.length - 1 ? C.blue : C.pale,
    line: i === flowLabels.length - 1 ? C.blue : C.pale,
    radius: 16,
  });
  addText(slide6, flowLabels[i], { left: flowXs[i] + 12, top: 248, width: 154, height: 44 }, {
    fontSize: 19, bold: true, color: i === flowLabels.length - 1 ? C.white : C.ink,
    alignment: "center", verticalAlignment: "middle",
  });
  if (i < flowLabels.length - 1) {
    slide6.shapes.add({
      geometry: "rightArrow",
      position: { left: flowXs[i] + 188, top: 255, width: 34, height: 28 },
      fill: C.cyan,
      line: { fill: "none", width: 0 },
    });
  }
}
addMetric(slide6, 56, "6类", "超导量子器件数据集", C.blue);
addMetric(slide6, 467, "≤ 5%", "正向与逆向预测误差", C.cyan);
addMetric(slide6, 878, "> 90%", "设计周期缩短", C.warm);
slide6.speakerNotes.textFrame.setText(
  "说明个人职责：参数化几何生成、批量仿真、数据整理与模型训练。六类器件包括低通和高通叉指电容、低通和高通螺旋电感、比特Z线及耦合器Z线。误差和周期缩短按简历口径展示，面试前建议再次确认测试集、误差定义和统计范围。涉及腾讯内部数据时只讲流程和方法，不展示未脱敏截图。"
);

const slide7 = presentation.slides.getItem(6);
reset(slide7);
addTitle(slide7, "科研成果与岗位匹配", 7);
addFooter(slide7, "王宇航｜感谢各位面试官");
addText(slide7, "代表性成果", { left: 58, top: 158, width: 500, height: 38 }, {
  fontSize: 24, bold: true, color: C.blue,
});
addBulletList(slide7, [
  "《基于参数化曲线的曲线OPC》｜光学学报，作者1",
  "Enforcement of curvy MRC compliance...｜Optics Express，作者2",
  "Parametric curvilinear OPC using B-splines...｜IWAPS，作者1",
  "参数化曲线掩模OPC方法｜发明专利，作者3",
], { left: 58, top: 218, width: 600, height: 260 }, { fontSize: 20, spaceAfterPoints: 11 });
slide7.shapes.add({
  geometry: "line",
  position: { left: 684, top: 170, width: 0, height: 390 },
  fill: "none",
  line: { style: "solid", fill: C.line, width: 1.5 },
});
addText(slide7, "岗位匹配", { left: 744, top: 158, width: 430, height: 38 }, {
  fontSize: 24, bold: true, color: C.blue,
});
const matches = [
  ["算法能力", "物理建模与数值优化"],
  ["工程能力", "Python验证与C++实现"],
  ["数据能力", "仿真自动化与预测模型"],
];
for (let i = 0; i < matches.length; i += 1) {
  const y = 226 + i * 98;
  addText(slide7, matches[i][0], { left: 744, top: y, width: 135, height: 32 }, {
    fontSize: 21, bold: true, color: C.ink,
  });
  addText(slide7, matches[i][1], { left: 892, top: y, width: 300, height: 48 }, {
    fontSize: 20, color: C.body,
  });
}
addCard(slide7, { left: 744, top: 510, width: 448, height: 92 }, { fill: C.pale, line: C.pale });
addText(slide7, "期待将这些经验应用到贵公司的算法研发工作中", { left: 770, top: 536, width: 396, height: 44 }, {
  fontSize: 20, bold: true, color: C.blue, alignment: "center",
});
addText(slide7, "谢谢", { left: 58, top: 532, width: 520, height: 70 }, {
  fontSize: 48, bold: true, color: C.ink,
});
slide7.speakerNotes.textFrame.setText(
  "结尾建议：我的研究经历主要集中在物理建模、数值优化和工程实现，也积累了仿真数据与模型训练经验。希望将这些能力应用到贵公司的算法研发工作中。以上是我的介绍，谢谢各位面试官。请根据具体岗位JD调整最后一句。"
);

await (await PresentationFile.exportPptx(presentation)).save(CANDIDATE);
for (let i = 0; i < presentation.slides.count; i += 1) {
  const preview = await presentation.slides.getItem(i).export({ format: "png", scale: 1.35 });
  await fs.writeFile(path.join(PREVIEW_DIR, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const snapshot = await presentation.inspect({
  kind: "slide,textbox,shape,image,notes,layout",
  include: "id,slide,name,title,textPreview,bbox,bboxUnit,alt",
  maxChars: 50000,
});
await fs.writeFile(path.join(BUILD_DIR, "candidate.inspect.ndjson"), snapshot.ndjson);
console.log(JSON.stringify({ candidate: CANDIDATE, previews: PREVIEW_DIR, slides: presentation.slides.count }));
