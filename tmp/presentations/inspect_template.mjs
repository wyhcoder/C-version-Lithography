import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "/Users/wangyuhang/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
console.log("presentation", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation)));
console.log("slides", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)));
console.log("slide0", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides.getItem(0))));
console.log("shapeCollection", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides.getItem(0).shapes)));
console.log("removeFn", presentation.slides.remove.toString().slice(0, 800));
console.log("slideDeleteFn", presentation.slides.getItem(0).delete.toString().slice(0, 800));
console.log((await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,chart,table,layout",
  include: "id,slide,name,title,textPreview,bbox,bboxUnit,isPlaceholder",
  maxChars: 30000,
})).ndjson);
