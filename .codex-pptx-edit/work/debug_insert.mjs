import {FileBlob, PresentationFile} from "@oai/artifact-tool";

const presentation = await PresentationFile.importPptx(await FileBlob.load("/Users/wangyuhang/Desktop/school/王宇航_技术面试汇报_初稿.pptx"));
const inserted = presentation.slides.insert({after: presentation.slides.getItem(4), layoutId: "/ppt/slideLayouts/slideLayout8.xml"});
console.log("inserted", inserted);
console.log("keys", inserted && Object.keys(inserted));
console.log("count", presentation.slides.items.length);
for (let index = 0; index < presentation.slides.items.length; ++index) {
    console.log(index, presentation.slides.getItem(index), Object.keys(presentation.slides.getItem(index)));
}
